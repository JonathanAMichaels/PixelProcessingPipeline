# Localizing NP detected spikes and registering raw data

## Installation
There are cython extensions !!!
```shell
pip install cython
python3 setup.py build_ext --inplace
pip install -e .
```

Warning: the `torch` installation may be trickly for newer GPUs that do not support the CUDA runtime 10.2. Please refer to the instructions on the PyTorch website if the install above provides error message related to `CUDA capability sm_86`. 

## This repository provides code for localizing the spikes detected in Neuropixels recordings, estimating motion from localization results, and tools for visualizing and evaluating the output of any spike sorter.

### Detection : 

To run detection and deduplication, and get a spike index (which is sufficient for localization, motion estimate and registration), the files contained in detect and the trained neural network weights detect.pt allow to run fast detection and deduplication of spikes on your standardized recording.

### Localization works as follow : 
 - It takes as input the spike index (times of detected spikes and main channels of these spikes) and filtered + standardized data
 - Spikes are read from the data and denoised by a Neural Network denoiser. Pre-trained weights are available on github. 
 The Pre-trained NN Denoiser model expects the spikes to be of temporal length 121 and aligned so that their minimum is reached at timestep 42. 
 After running detection, make sure that spike times are aligned in the sense that all spike minimum are attained at the same timestep. In localization_pipeline/run.py, we've set up `detector_min = 60`. This timestep corresponds to the minimum of the spikes obtained with voltage threshold detector.
 It is important to check and input the correct minimum time before running localization, for proper denoising. 
 - Spikes are then localized for each batch of data (for example each second of data), obtained positions and features are stored into the desired repository before being merged to give final results. 
 
Localization code is designed to be fully self-contained. Code to read data and denoise data is written following the YASS pipeline (YASS: Yet Another Spike Sorter applied to large-scale multi-electrode array recordings in primate retina, Lee et al., 2020) and github repository (https://github.com/paninski-lab/yass).
Instructions to train and obtain a new NN-Denoiser can be found on YASS github repository. 

If you have ran clustering and obtained a spike train, then it is possible to run run_with_residuals.py. This method takes advantage of template information.  

### Motion estimate works as follow : 
 - It takes as input the localizations/amplitudes/spike times/geometry array of spikes.
 - A raster plot is generated from the inputs, and it is destriped/denoised.
 - By default, non-rigid image-based decentralized registration is run, and the
   motion estimate + original raster + registered raster are saved.

### Visualisation : 

The repository contain a script that provides a 3d interactive visualization of the spike train and its localization features. 
Code requires Datoviz (https://github.com/datoviz/datoviz.git, see documentation here https://datoviz.org) 

### Quality Metrics : 

The files in quality_metric show how to compute simple metrics for quantitatively evaluating the quality of registration or the quality of the spike train. 
drift_metrics.py is a script containing code to derive a general correlation score (taken from A New Coefficient of Correlation, Sourav Chatterjee, 2020) between the amplitude (PTP) of units and displacement estimate, firing rate and displacement estimates, and between the first PC of neural activity and displacement. 
The first two scores allow to evaluate the quality of individual units (ideally, a good unit would have low FR and PTP correlation with displacement) and the quality of the recording (which should have low correlation between the first PC of neural activity and displacement). 
