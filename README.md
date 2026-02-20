<div align="center">
  
# Simulated CAsual Representations in medical images (SCAR)

</div>
</p>

### Abstract 
Causally linking disease-related factors to image-derived biomarkers provides a powerful pathway to understanding disease mechanisms. Despite growing interest in applying causal artificial-intelligence (AI) approaches for this task, these methods still need to be adapted for complex medical images, and especially, neuroimaging. However, the lack of ground-truth data presents a barrier to development.  

To bridge this gap, we developed and tested a method for generating synthetic neuroimages, which adhere to a user-specified causal structure describing the non-image to image variable relationships, permitting the creation of ground-truth neuroimaging datasets. In the simulated T1-weighted magnetic resonance images, anatomical variability is modeled by sampling from a subspace estimated from real data and deforming a template image to create unique simulated subjects. Causal relationships are encoded via precise volumetric changes of any region-of-interest without unwanted global artifacts.  

We achieved relative volume errors of 0.3-2.66% for the targeted regions-of-interest and demonstrate their statistically significant causal relationships, while maintaining mean absolute errors for non-target brain regions between 0.034-0.397ml. An initial evaluation of causal discovery methods exposes their limited ability to suppress spurious connections, highlighting the need for image-appropriate methods.  

Our framework is the first to enable the generation of realistic synthetic 3D neuroimages with explicit causal control that can serve as the missing ground-truth data necessary for the objective benchmarking and development of causal AI methods. 


## Generating Data

### Generative models for sampling effects
We use PCA models fit to velocity fields (Log-Euclidian framework) derived from 50 T1-weighted brain MRIs ([IXI](https://brain-development.org/ixi-dataset/) dataset) nonlinearly registered to the [SRI24 atlas](https://www.nitrc.org/projects/sri24) as generative models for sampling global subject effects. The code used for generating these models is in: 
```bash
├── pca
│   ├── pca_isv_velo_ixi.py
│   ├── pca_velo_ixi.py
│   ├── save_roi_masks.py
│   └── subspacemodels.py
```
* `pca_isv_velo_ixi.py` fits the PCA model for the full subject morphology.
* `pca_velo_ixi.py` fits the PCA models for the localized region morphology.

Regions-of-intered are defined by SynthSeg segmentations and mask outputs. 

Example shell script for generating PCA models in `example_pca.sh`.

### Dataset generatio
Subject variability is represented by morphological variation introduced to a template image (we use the SRI24 atlas). Causal effects are represented by increasing or decreasing a target roi's volume as specified by the user defined SCM. Generating synthetic datasets requires the following files: 
```bash
├── generate_data
│   ├── define_causal_graph.py
│   ├── generate_causal_data.py
│	├── volume_control.py
│   ├── kernels.py
│   ├── subspacemodels.py
│   ├── pca_utils.py
│   └── utils.py
```

* `define_causal_graph.py` is where the user can define the causal structures they want the data to adhere to. Define your causal variable distributions and the SCM to define your effects. Inter-subject variablilty distrbutions are also defined.
* `generate_causal_data.py` generates the data as specified by the SCM in `.nii.gz` format.

Example shell script for generating data in `example_generate_causal_data.sh`.
**NOTE: the target ROI is defined here (not in define_causal_graph.py)**

## Environment 
Our dataset generation code used:
* Python 3.10.6
* simpleitk 2.1.1.1
* antspyx 0.3.4
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* scipy 1.9.1

Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.

* Questions? Open an issue or send an [email](mailto:eryn.libertscott@ucalgary.ca?subject=SCAR).

