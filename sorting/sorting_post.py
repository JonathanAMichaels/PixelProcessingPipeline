import spikeinterface.full as si
import numpy as np
from pathlib import Path
from spikeinterface.preprocessing import highpass_spatial_filter
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
from spikeinterface.preprocessing.motion import load_motion_info
from spikeinterface.preprocessing.normalize_scale import scale
from spikeinterface.sorters import Kilosort2_5Sorter
import shutil
import subprocess

def sorting_post(config):
    dataset_folder = Path(config['neuropixel_folder'])
    sorting_folder = dataset_folder / 'kilosort2.5'
    waveform_folder = sorting_folder / 'waveforms_kilosort2.5'
    we = si.WaveformExtractor.load(waveform_folder)

    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation',
                                                           'amplitude_median', 'amplitude_cutoff', 'silhouette'])
    metrics.to_csv(sorting_folder / 'metrics.csv')

