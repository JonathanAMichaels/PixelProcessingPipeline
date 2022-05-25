import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import parmap

from detector import Detect
from localization_pipeline.denoiser import Denoise

from deduplication import deduplicate_gpu, deduplicate

from scipy.signal import argrelmin

from run import run

## ADD ARGUMENTS
geom_path = 'spikes_localization_registration/channels_maps/np2_channel_map.npy'
path_nn_detector = 'spikes_localization_registration/pretrained_detector/detect.pt'
path_nn_denoiser = 'spikes_localization_registration/pretrained_denoiser/denoise.pt'
standardized_path = 'standardized.bin'
standardized_dtype = 'float32'
sampling_rate = 30000
len_recording = 1000
output_directory = 'detection_results'

geom_array = np.load(geom_path)
apply_nn = True ### If set to false, run voltage threshold instead of NN detector 
spatial_radius = 70
n_sec_chunk = 1
n_processors = 4
n_sec_chunk_gpu_detect = .1
detect_threshold = 0.5 ## 0.5 if apply NN, 4/5/6 otherwise 
n_filters_detect = [16, 8, 8] 
spike_size_nn = 4 ### In ms
n_filters_denoise = [16, 8, 4]
filter_sizes_denoise = [5, 11, 21]

run(standardized_path, standardized_dtype, output_directory, geom_array, spatial_radius, apply_nn, n_sec_chunk, n_batches, n_processors, n_sec_chunk_gpu_detect, sampling_rate, len_recording,
    detect_threshold, path_nn_detector, n_filters_detect, spike_size_nn, path_nn_denoiser, n_filters_denoise, filter_sizes_denoise, run_chunk_sec='full')