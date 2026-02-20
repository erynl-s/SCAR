'''
author: eryn
adapted from emma's 'generate_data.py'

'''

import time
import ants
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import pandas as pd
import utils
from numpy.random import Generator, PCG64
from volume_control import posterior_channel_for_label, extract_channel_3d, build_v0_from_sdf, compute_flux, warp_mask_with_svf, bracket_around_s0, f, scale_vector_image, soft_volume_of_s
import roi_label_map

# ----------------------------------------
#                arguments
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--t_rois', nargs="+", help='name of target ROIs to adjust volume of', required=True)
parser.add_argument('--c_graph', type=str, help='causal graph structure for data gen', required=True)
parser.add_argument('--isv', type=int, help='1 to add intersubject variability with real displacement fields, else 0', required=True)
parser.add_argument('--effect', type=int, help='1 to add disease effect with defined volume change, else 0', required=True)
parser.add_argument('--expname', type=str, required=True)
args = parser.parse_args()


# ----------------------------------------
#                  setup
# ----------------------------------------
main_dir = '/home/eryn/SimBA/'

# extract target ROI names from arguments
target_regions = args.t_rois
c_graph = args.c_graph
#create directory for dataset

save_dir = main_dir + args.expname + "/" + c_graph + "/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# read template image (3d)
fpath = '/home/eryn/SimBA/sri24_spgr_RAI_ROI.nii.gz' #atlas with RAI orientation + origin set to 0,0,0 to be consistent with itk + bspline method
itk_img = utils.load_image(fpath)
#clip intensities to remove outliers
itk_array = sitk.GetArrayFromImage(itk_img)
itk_array = np.clip(itk_array, a_min=0, a_max=800)
itk_img = sitk.GetImageFromArray(itk_array)
itk_img.CopyInformation(utils.load_image(fpath))
# Add intensity normalization
itk_img = utils.intensity_normalization(itk_img)

# read label atlas
lpath = '/home/eryn/SynthSegOutput/sri24_spgr_RAI_ROI/sri24_spgr_RAI_ROI_seg.nii.gz' #/home/eryn/SimBA/sri24_spgr_lpba40_labels_RAI_ROI.nii.gz' #path to label atlas
label_atlas = sitk.ReadImage(lpath)

ppath = '/home/eryn/SynthSegOutput/sri24_spgr_RAI_ROI/posteriors.nii.gz'
post_atlas = sitk.ReadImage(ppath)

#create itk objects for later on
writer = sitk.ImageFileWriter()
warper_bspline = sitk.WarpImageFilter()
warper_linear = sitk.WarpImageFilter()
warper_nn = sitk.WarpImageFilter()
warper_linear.SetInterpolator(sitk.sitkLinear)
warper_bspline.SetInterpolator(sitk.sitkBSpline)
warper_nn.SetInterpolator(sitk.sitkNearestNeighbor)

# load distribution information based on split
dataframe = pd.read_csv('/home/eryn/SimBA/' + args.expname + '/effect_distributions/' + 'train_dst.csv', index_col=0)

# extract effect and isv distribution values
effect1_dst = dataframe['effect1_dst'].values
# effect2_dst = dataframe['effect2_dst'].values
age_dst = dataframe['a_dst'].values
edu_dst = dataframe['b_dst'].values
isv_dst = dataframe['isv_dst'].values

n_samples = len(dataframe)


SEED=len(dataframe) #seed for generating effect distribution
numpy_randomGen = Generator(PCG64(SEED))

# ----------------------------------------
#           effect models (PCA)
# ----------------------------------------
# load required PCA models and define sampling distribution along components
pca_dir = '/home/eryn/SimBA/save_pca_synthseg/pca_models_velo_50/' #path to pca model directory


# ------ inter-subject variability effects ------
if args.isv == 1:
    pca_isv = utils.load_pca_model(pca_dir, 'sri24_spgr_RAI_synthseg_isv') #open PCA model for subject effects
    n_comps_isv = 10 #number of PCA components to sample for subject effects



# ----------------------------------------
#           data generation
# ----------------------------------------

def secant_solve(f, s_lo, s_hi, tol_abs, tol_rel, max_it=20):
    f_lo, f_hi = f(s_lo,v0,chan_boxed,Vt), f(s_hi,v0,chan_boxed,Vt)
    for _ in range(max_it):
        if abs(f_hi - f_lo) < 1e-12:  # fallback to bisection
            s_mid = 0.5*(s_lo+s_hi)
        else:
            s_mid = s_hi - f_hi*(s_hi - s_lo)/(f_hi - f_lo)
        fm = f(s_mid,v0,chan_boxed,Vt)
        # bracket maintenance
        if f_lo*fm <= 0: s_hi, f_hi = s_mid, fm
        else:            s_lo, f_lo = s_mid, fm
        # convergence
        if abs(fm) <= max(tol_abs, tol_rel*max(1.0, abs(f_hi), abs(f_lo))): 
            return s_mid
    return 0.5*(s_lo+s_hi) 

fname_lst = []
fpath_lst = []
original_volumes = []
target_volumes = []
achieved_volumes = []
generated_images = []

start_time = time.time()

print('generating data')
for n in range(n_samples):
    print(n)
    #get label for image
    if len(target_regions) == 1:
        label = str(np.around(isv_dst[n],3)) + '_S_'  + str(np.around(age_dst[n],3)) + '_A_' + str(np.around(effect1_dst[n],3)) + '_' + str(target_regions[0]) #+ str(np.around(edu_dst[n],3)) + '_B'# +"_" + str(np.around(effect2_dst[n],3)) + '_' + 
    else:
        label = str(np.around(isv_dst[n],3)) + '_S_'  + str(np.around(age_dst[n],3)) + '_A_' + str(np.around(effect1_dst[n],3)) + '_' + str(target_regions[0]) + '_'  #str(np.around(effect2_dst[n],3)) + '_' + str(target_regions[1]) #+ str(np.around(edu_dst[n],3)) + '_B_' +"_" +
    # ---------- add data information to dataframe columns ------------
    dataframe.loc[dataframe.index[n], 'filepath'] = save_dir + str(n).zfill(5) + '_' + label + '.nii.gz'
    dataframe.loc[dataframe.index[n], 'filename'] = str(n).zfill(5) + '_' + label + '.nii.gz'
    fname_lst.append(str(n).zfill(5) + '_' + label + '.nii.gz')
    fpath_lst.append(save_dir + str(n).zfill(5) + '_' + label + '.nii.gz')

    # ---------------------------------------------------------------
    warper_bspline.SetOutputParameteresFromImage(itk_img)


    # ------- include inter-subject variability effects -------
    if args.isv == 1:
        #sample (fwd) velocity field from first n components isv effect pca mode
        print('isv sampling val: {}'.format(isv_dst[n]))
        vf_isv_np = pca_isv.translation_vector + pca_isv.basis[:,:n_comps_isv]*np.sqrt(pca_isv.eigenvalues[:n_comps_isv])*isv_dst[n] #subject velocity field as np arr
        vf_isv = utils.pca_vector_to_itk_full(vf_isv_np, itk_img) #subject velocity field as sitk image type

        #perform scaling and squaring on velocity field to get displacement field for the subject effect
        #need inverse (atlas -> subject) disp fields for generating "subject" template
        inv_df_isv = utils.svf_scaling_and_squaring(vf_isv, compute_inverse=True)

        #generate subject image
        s_img = warper_bspline.Execute(itk_img, inv_df_isv)
 

    else:
        s_img = itk_img

    if args.effect == 1:
        for roi in target_regions:                
            # -------- get masks for roi-------
            print("making changes to roi:", roi)
            t_roi_numerical_label = roi_label_map.get_numerical_label(roi)
            troi_mask = utils.get_mask(label_atlas, t_roi_numerical_label)
            # get the posterior for the current label
            k, counts = posterior_channel_for_label(label_atlas, post_atlas, t_roi_numerical_label)
            chan = extract_channel_3d(post_atlas, k)

            warper_linear.SetOutputParameteresFromImage(chan)
            warper_nn.SetOutputParameteresFromImage(troi_mask)

            if args.isv == 1:
                #----- apply isv changes to masks-------
                chan = warper_linear.Execute(chan, inv_df_isv)
                troi_mask = warper_nn.Execute(troi_mask, inv_df_isv)

            if roi == target_regions[0]:
                effect_dst = effect1_dst
            else:
                effect_dst = effect2_dst
        # ------- volume change effects -------
            # sample (forward) velocity field from first component in disease effect pca model
            #print('d sampling val: {}'.format(effect_dst[n]))
            

            # use the warped posterior for the volume change, get the velo field for the defined vol change and roi
            # get vol of posterior
            # chan to array

            def bbox_from_mask(mask, pad_mm):
                ls = sitk.LabelShapeStatisticsImageFilter()
                ls.Execute(sitk.Cast(mask>0, sitk.sitkUInt8))
                x,y,z,sx,sy,sz = ls.GetBoundingBox(1)
                sp = np.array(mask.GetSpacing())
                pad = np.maximum(np.round(pad_mm / sp), 2).astype(int)
                start = np.maximum([x-pad[0],y-pad[1],z-pad[2]], 0)
                size  = np.minimum([sx+2*pad[0], sy+2*pad[1], sz+2*pad[2]],
                                    np.array(mask.GetSize())-start)
                return tuple(map(int,start)), tuple(map(int,size))

            def extract_roi(im, start, size):
                return sitk.Extract(im, size=size, index=start)

            # before building v0:
            start,size = bbox_from_mask(troi_mask, pad_mm=6.0) 
            mask_boxed   = extract_roi(troi_mask, start,size)
            chan_boxed   = extract_roi(chan, start,size)


            v0     = build_v0_from_sdf(mask_boxed)
            Phi1   = compute_flux(v0, mask_boxed)
            V0 = sitk.GetArrayFromImage(chan_boxed).sum(dtype=np.float64)
            # V0   = soft_volume_of_s(0.0, v0, chan_boxed)
            # eps  = 0.5
            # slope= (soft_volume_of_s(+eps, v0, chan_boxed) - soft_volume_of_s(-eps, v0, chan_boxed)) / (2*eps)

            # bracket around s0
            Vt = float(effect_dst[n]) * V0
            print("Initial volume:", V0, " Target volume:", Vt, "effect:", effect_dst[n])
            s0     = (Vt - V0) / Phi1 
            # s0   = (Vt - V0) / max(slope, 1e-6)
            print("Initial guess for s0:", s0)
            s_lo, s_hi = bracket_around_s0(f, s0, v0, chan_boxed, V0, Vt)

            s_star= secant_solve(f, s_lo, s_hi, tol_abs=1.0, tol_rel=1e-4)

            v = build_v0_from_sdf(troi_mask)

            v_final = scale_vector_image(v, s_star)

            #perform scaling and squaring on velocity field to get displacement field
            #get the inverse field since that's what we want to apply to the image
            df = utils.svf_scaling_and_squaring(v_final, compute_inverse=True)

            #apply transformed effect to subject
            gen_img=warper_bspline.Execute(s_img, df)

            s_img = gen_img # so that next ROI we apply changes to the already changed image

            # add original volume to a list
            # original_volumes.append(V0)
            # target_volumes.append(Vt)
            # achieved_volumes.append(soft_volume_of_s(s_star,v0, chan))
            n_time = time.time()
            print(f"processing time for sample {n}: {n_time - start_time:.2f} seconds")
    else:
        gen_img = s_img


    # ------- save generated image -------
    #clip intensities to remove outliers
    gen_array = sitk.GetArrayFromImage(gen_img)
    gen_array = np.clip(gen_array, a_min=0, a_max=800)
    gen_img = sitk.GetImageFromArray(gen_array)
    gen_img.CopyInformation(itk_img)
    # normalize intensities to 0-1 range before saving
    gen_img = utils.intensity_normalization(gen_img)
    print('saving sample {}'.format(n))

    writer.SetFileName(save_dir + str(n).zfill(5) + '_' + label + '.nii.gz')
    writer.Execute(gen_img)


end_time = time.time()
print(f"Total processing time for {n} samples: {end_time - start_time:.2f} seconds")

# ---------------- save dataframe ---------------------

dataframe['filepath']=fpath_lst
dataframe['filename']=fname_lst
# if args.effect == 1:
#     dataframe['original_volume']=original_volumes
#     dataframe['target_volume']=target_volumes
#     dataframe['achieved_volume']=achieved_volumes

dataframe.to_csv(save_dir + args.expname + '.csv')
