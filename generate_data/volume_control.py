'''
author: eryn
'''
import ants
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import pandas as pd
import utils
from numpy.random import Generator, PCG64
import roi_label_map
import nibabel as nib

# ----------------------------------------
#                  setup
# ----------------------------------------
main_dir = '/home/eryn/SimBA/'

# read template image (3d)
fpath = '/home/eryn/SimBA/sri24_spgr_RAI_ROI.nii.gz' #atlas with RAI orientation + origin set to 0,0,0 to be consistent with itk + bspline method
itk_img = utils.load_image(fpath)


# read label atlas
lpath = '/home/eryn/SynthSegOutput/sri24_spgr_RAI_ROI/sri24_spgr_RAI_ROI_seg.nii.gz' #path to label atlas
atlas = sitk.ReadImage(lpath)

spath = '/home/eryn/SynthSegOutput/sri24_spgr_RAI_ROI/posteriors.nii.gz' # path to posteriors
soft_atlas = sitk.ReadImage(spath)

# make a copy of atlas
label_atlas = sitk.Image(atlas)
label_atlas.CopyInformation(atlas)
# read atlas volumes
atlas_volumes = pd.read_csv('/home/eryn/SynthSegOutput/sri24_spgr_RAI_ROI/volumes.csv', index_col=0)

#create itk objects for later on
writer = sitk.ImageFileWriter()
warper = sitk.WarpImageFilter()
warper.SetInterpolator(sitk.sitkBSpline)
warper.SetOutputParameteresFromImage(itk_img)
warper_linear = sitk.WarpImageFilter()
warper_linear.SetInterpolator(sitk.sitkLinear)
warper_linear.SetOutputParameteresFromImage(itk_img)
warper_nn = sitk.WarpImageFilter()
warper_nn.SetInterpolator(sitk.sitkNearestNeighbor)
warper_nn.SetOutputParameteresFromImage(itk_img)

#####################################

# get SDF of target mask 
def get_SDF(mask):
    # note sign convention is outside is positive, when adjusting volume expand will be default, shrink will be sign negative
    sdf = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False, useImageSpacing=True)
    return sdf


def neighbour_labels_for_target(label_atlas_sitk, target_label):
    # atlas as numpy
    atlas = sitk.GetArrayFromImage(label_atlas_sitk)

    # target mask (numpy)
    tgt = (atlas == target_label).astype(np.uint8)
    tgt_img = sitk.GetImageFromArray(tgt)
    tgt_img.CopyInformation(label_atlas_sitk)

    # dilate once around the target
    dil = sitk.BinaryDilate(tgt_img)

    dil_np = sitk.GetArrayFromImage(dil).astype(bool)

    # look up labels in the dilated band
    neigh = atlas[dil_np]

    # unique labels, excluding background and the target itself
    labels = np.unique(neigh)
    labels = labels[(labels != 0) & (labels != target_label)]

    return labels.tolist()

# get surface normal 
def get_boundary_normal(region_mask):
    sdf = get_SDF(region_mask)
    gf = sitk.GradientImageFilter()
    gf.SetUseImageSpacing(True)
    grad  = gf.Execute(sdf)  # VectorImage

    # get direction components from gradient vector
    gx = sitk.VectorIndexSelectionCast(grad, 0)
    gy = sitk.VectorIndexSelectionCast(grad, 1)
    gz = sitk.VectorIndexSelectionCast(grad, 2)

    mag = sitk.Sqrt(gx*gx + gy*gy + gz*gz)
    mag = sitk.Maximum(mag, 1e-12)  # eps_norm

    # normals of region
    n = sitk.Compose(gx/mag, gy/mag, gz/mag)
    n.CopyInformation(region_mask)

    return n



def build_v0_from_sdf(mask: sitk.Image, sigma_mm: float = 1.5, band_mm: float = 2.0) -> sitk.Image:
    to64 = lambda im: sitk.Cast(im, sitk.sitkFloat64)

    # SDF (inside negative), force Float64
    phi = sitk.SignedMaurerDistanceMap(mask, insideIsPositive=False,
                                            squaredDistance=False, useImageSpacing=True)

    # Unit normals n = ∇φ / |∇φ| (spacing-aware), all Float64
    gf = sitk.GradientImageFilter(); gf.SetUseImageSpacing(True)
    g  = gf.Execute(phi)                                        # vector image (Float64)
    gx = sitk.VectorIndexSelectionCast(g, 0)
    gy = sitk.VectorIndexSelectionCast(g, 1)
    gz = sitk.VectorIndexSelectionCast(g, 2)
    mag = sitk.Maximum(sitk.Sqrt(gx*gx + gy*gy + gz*gz), 1e-12) # Float64
    nx, ny, nz = gx/mag, gy/mag, gz/mag                         # Float64

    # Two-sided Gaussian falloff ψ(φ) centered at surface
    t = phi/sigma_mm
    psi    = sitk.Exp(-0.5*(t*t))
    psi.CopyInformation(mask)

    # Hard band |φ| ≤ band_mm 
    band = sitk.Cast(sitk.Abs(phi) <= band_mm, sitk.sitkFloat64)

    # Localized field v0 = ψ * band * n  
    vx = nx * psi * band
    vy = ny * psi * band
    vz = nz * psi * band

    v0 = sitk.Compose(vx, vy, vz)                               # vector image
    v0.CopyInformation(mask)
    return sitk.Cast(v0, sitk.sitkVectorFloat64)                



# compute flux using divergence theorem
def compute_flux(velo_field, region_mask):
    # get components of velocity field
    vx = sitk.VectorIndexSelectionCast(velo_field, 0)
    vy = sitk.VectorIndexSelectionCast(velo_field, 1)
    vz = sitk.VectorIndexSelectionCast(velo_field, 2)

    gf = sitk.GradientImageFilter(); gf.SetUseImageSpacing(True)
    dvx = gf.Execute(vx); dvxdx = sitk.VectorIndexSelectionCast(dvx,0)
    dvy = gf.Execute(vy); dvydy = sitk.VectorIndexSelectionCast(dvy,1)
    dvz = gf.Execute(vz); dvzdz = sitk.VectorIndexSelectionCast(dvz,2)

    div = dvxdx + dvydy + dvzdz
    div_np = sitk.GetArrayFromImage(div).astype(np.float64)
    div_np = np.nan_to_num(div_np, nan=0.0, posinf=0.0, neginf=0.0)

    m_np   = sitk.GetArrayFromImage(region_mask).astype(bool)
    #print("Divergence in mask:", div_np[m_np])
    voxel_vol = np.prod(region_mask.GetSpacing()) # this equals 1, not sure if this is correct or what it should be
    return float(div_np[m_np].sum() * voxel_vol)


def scale_vector_image(v, s):
    comps = [sitk.VectorIndexSelectionCast(v,i)*s for i in range(3)]
    out = sitk.Compose(*comps) 
    out.CopyInformation(v) 
    return out

def warp_mask_with_svf(mask: sitk.Image, v: sitk.Image, accuracy: int = 4) -> sitk.Image:
    disp = utils.svf_scaling_and_squaring(v, accuracy=accuracy, compute_inverse=True)
    warped = warper_nn.Execute(mask, disp)

    return warped

def volume_vox(mask: sitk.Image) -> float:
    return float(sitk.GetArrayFromImage(mask).astype(bool).sum())

def volume_of_s(s: float, v0: sitk.Image, roi_mask: sitk.Image, acc: int = 4) -> float:
    v_scaled = scale_vector_image(v0, s)
    warped   = warp_mask_with_svf(roi_mask, v_scaled)
    return float(sitk.GetArrayFromImage(warped).astype(bool).sum())

# given f(s) = V(s) - Vt
def bracket_around_s0(f, s0, v0, chan, V0, Vt, expand_factor=2.0, max_expand=20):
    '''
    Create a bracket around s0 where f(s) changes sign.
    param f: function to evaluate
    param s0: initial guess for root
    param v0: initial volume
    param k: current iteration
    param soft_atlas: soft atlas image
    param V0: initial target volume
    param Vt: target volume
    param expand_factor: factor by which to expand the bracket
    param max_expand: maximum number of expansions
    '''
    # initial bracket: (negative, 0] for shrink; [0, positive) for grow
    if Vt < V0:       # shrink
        s_lo, s_hi = (min(-abs(s0), -1e-3), 0.0)
    else:             # grow
        s_lo, s_hi = (0.0, max(abs(s0),  1e-3))

    # volume of s hi and lo
    f_lo, f_hi = f(s_lo, v0, chan, Vt), f(s_hi, v0, chan, Vt)
    for _ in range(max_expand):
        if f_lo * f_hi <= 0:             # straddles the root
            return s_lo, s_hi
        # Expand the endpoint that is **not** zero, in the direction of s0
        if Vt < V0:                       # shrink → push s_lo more negative
            s_lo *= expand_factor
            f_lo  = f(s_lo, v0, chan, Vt)
        else:                             # grow → push s_hi more positive
            s_hi *= expand_factor
            f_hi  = f(s_hi, v0, chan, Vt)
    raise RuntimeError("Could not bracket root; check flux sign and warp effect.")


def posterior_channel_for_label(seg_img: sitk.Image, post_img: sitk.Image, label_val: int):
    seg = sitk.GetArrayFromImage(seg_img)           # (Z,Y,X)
    post = sitk.GetArrayFromImage(post_img)         # (Z,Y,X,K)
    winners = np.argmax(post, axis=0)              # (Z,Y,X) argmax channel per voxel
    m = (seg == label_val)
    # which posterior channel most often "wins" inside this label?
    counts = np.bincount(winners[m].ravel(), minlength=post.shape[-1])
    k = int(counts.argmax())
    return k, counts

def extract_channel_3d(post4d: sitk.Image, k: int) -> sitk.Image:
    size  = list(post4d.GetSize())  # [X,Y,Z,K]
    idx   = [0,0,0,0]; idx[3]=k
    size[3]=0
    return sitk.Extract(post4d, size, idx)  # 3D float

def soft_vol_from_warped_chan(chan, disp_inv):

    ch_warp = warper_linear.Execute(chan, disp_inv)
    arr = sitk.GetArrayFromImage(ch_warp)                    # (Z,Y,X)
    vx  = float(np.prod(ch_warp.GetSpacing()))             
    return float(arr.sum(dtype=np.float64) * vx)

def soft_volume_of_s(s: float, v, chan) -> float:
    v_scaled = scale_vector_image(v, s)
    disp_inv = utils.svf_scaling_and_squaring(v_scaled, accuracy=4, compute_inverse=True)
    return soft_vol_from_warped_chan(chan, disp_inv)

def f(s,v, chan, Vt): return soft_volume_of_s(s, v, chan) - Vt
