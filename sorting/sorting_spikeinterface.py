import spikeinterface.full as si
import numpy as np
from pathlib import Path
from spikeinterface.preprocessing import highpass_spatial_filter
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
from spikeinterface.preprocessing.motion import load_motion_info
from spikeinterface.preprocessing.normalize_scale import scale
from spikeinterface.sorters import Kilosort2_5Sorter
import shutil

import shutil
import time
import os
import errno

def is_dir_locked(directory):
    """Check if any file within the directory is locked/open."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try to open the file in append mode and close it immediately
                with open(file_path, 'a'):
                    pass
            except IOError as e:
                # If file is in use, it will throw an IOError
                return True
    return False

def robust_rmtree(path, max_retries=5, delay=1):
    """Attempt to delete a directory tree multiple times, waiting between retries."""
    for i in range(max_retries):
        if is_dir_locked(path):
            print("Directory is locked or files are in use, waiting to retry...")
            time.sleep(delay)
            continue

        try:
            shutil.rmtree(path)
            print("Directory successfully deleted.")
            break
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                # Directory not empty
                print("Directory not empty, retrying...")
                time.sleep(delay)
            else:
                raise



def sorting(config):
    dataset_folder = Path(config['neuropixel_folder'])
    motion_folder = dataset_folder / 'motion'
    sorting_folder = dataset_folder / 'kilosort2.5_new'
    waveform_folder = sorting_folder / 'waveforms_kilosort2.5'
    if sorting_folder.exists() and sorting_folder.is_dir():
        #shutil.rmtree(sorting_folder)
        robust_rmtree(sorting_folder)

    spikeglx_folder = dataset_folder
    # global kwargs for parallel computing
    job_kwargs = dict(
        n_jobs=-1,
        chunk_duration='1s',
        progress_bar=True,
    )
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    print(stream_names)

    raw_rec = si.read_spikeglx(spikeglx_folder, stream_name=stream_names[0], load_sync_channel=False)
    rec_eval_noise = si.highpass_filter(recording=raw_rec, freq_min=400.)
    bad_channel_ids1, channel_labels = si.detect_bad_channels(rec_eval_noise, method='mad', std_mad_threshold=1.5,
                                                              chunk_duration_s=0.3,
                                                              num_random_chunks=100)
    bad_channel_ids2, channel_labels = si.detect_bad_channels(rec_eval_noise,
                                                              chunk_duration_s=0.3,
                                                              num_random_chunks=100)
    bad_channel_ids = np.concatenate((bad_channel_ids1, bad_channel_ids2))
    print(bad_channel_ids)

    rec1 = raw_rec.remove_channels(bad_channel_ids)
    rec1 = scale(rec1, 20)
    rec1 = si.bandpass_filter(recording=rec1, freq_min=300., freq_max=10000.)
    rec1 = si.phase_shift(rec1)
    rec1 = highpass_spatial_filter(rec1)

    motion_info = load_motion_info(motion_folder)
    rec_corrected = interpolate_motion(
        recording=rec1,
        motion=motion_info['motion'],
        temporal_bins=motion_info['temporal_bins'],
        spatial_bins=motion_info['spatial_bins'],
        **motion_info['parameters']['interpolate_motion_kwargs'])

    params_kilosort2_5 = si.get_default_sorter_params('kilosort2_5')
    params_kilosort2_5['do_correction'] = False
    params_kilosort2_5['skip_kilosort_preprocessing'] = False
    params_kilosort2_5['scaleproc'] = 50
    print(params_kilosort2_5)
    Kilosort2_5Sorter.set_kilosort2_5_path('sorting/Kilosort-2.5')
    sorting = si.run_sorter('kilosort2_5', rec_corrected, output_folder=str(sorting_folder),
                            verbose=True, remove_existing_folder=True, **params_kilosort2_5)

    we = si.extract_waveforms(rec_corrected, sorting, folder=str(waveform_folder),
                              sparse=True, max_spikes_per_unit=500, ms_before=1.5, ms_after=2.,
                              **job_kwargs)

    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                           'isi_violation', 'amplitude_cutoff'])
    metrics