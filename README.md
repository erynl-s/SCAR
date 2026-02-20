<div align="center">
  
# Simulated CAsual Representations in medical images (SCAR)

</div>
</p>


Implementation for the SimBA 🦁 framework presented and utilized in our papers: 
* [A flexible framework for simulating and evaluating biases in deep learning-based medical image analysis](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_46) (MICCAI 2023)
* [Towards objective and systematic evaluation of bias in artificial intelligence for medical imaging](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocae165/7701447) (JAMIA 2024)
* [Where, why, and how is bias learned in medical image analysis models? A study of bias encoding within convolutional networks using synthetic data](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(24)00537-1/fulltext) (eBioMedicine 2025)

Our code here is based on our initial feasibility study of spatially localized morphological bias effects in structural neuroimaging datasets. However, the crux of the SimBA framework is the **systematic augmentation of a template image with disease (target) effects, bias effects, and subject effects**. This simple procedure can be expanded to other organ templates and imaging modalities. 


### Abstract 
Despite the remarkable advances in deep learning for medical image analysis, it has become evident that biases in datasets used for training such models pose considerable challenges for a clinical deployment, including fairness and domain generalization issues. Although the development of bias mitigation techniques has become ubiquitous, the nature of inherent and unknown biases in real-world medical image data prevents a comprehensive understanding of algorithmic bias when developing deep learning models and bias mitigation methods. To address this challenge, we propose a modular and customizable framework for bias simulation in synthetic but realistic medical imaging data. Our framework provides complete control and flexibility for simulating a range of bias scenarios that can lead to undesired model performance and shortcut learning. In this work, we demonstrate how this framework can be used to simulate morphological biases in neuroimaging data for disease classification with a convolutional neural network as a first feasibility analysis. Using this case example, we show how the proportion of bias in the disease class and proximity between disease and bias regions can affect model performance and explainability results. The proposed framework provides the opportunity to objectively and comprehensively study how biases in medical image data affect deep learning pipelines, which will facilitate a better understanding of how to responsibly develop models and bias mitigation methods for clinical use.


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

