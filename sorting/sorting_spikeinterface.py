import spikeinterface.full as si
import numpy as np
from pathlib import Path
from spikeinterface.preprocessing import highpass_spatial_filter
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
from spikeinterface.preprocessing.motion import load_motion_info
from spikeinterface.preprocessing.normalize_scale import scale
from spikeinterface.sorters import Kilosort2Sorter
import shutil
import subprocess

def unlock_files(directory):
    # Find the process IDs using the files in the directory
    process_ids = set()
    result = subprocess.run(['lsof', '+D', directory], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[1].isdigit():
            process_ids.add(parts[1])

    # Kill the processes
    for pid in process_ids:
        print(f"Killing process {pid}")
        subprocess.run(['kill', '-9', pid])

def sorting(config):
    dataset_folder = Path(config['neuropixel_folder'])
    motion_folder = dataset_folder / 'motion'
    sorting_folder = dataset_folder / 'kilosort2.0'
    waveform_folder = sorting_folder / 'waveforms_kilosort2.0'
    if sorting_folder.exists() and sorting_folder.is_dir():
        unlock_files(sorting_folder)
        shutil.rmtree(sorting_folder)

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
    #rec1 = scale(rec1, 20)
    rec1 = si.bandpass_filter(recording=rec1, freq_min=300., freq_max=10000.)
    rec1 = si.phase_shift(rec1)
    rec1 = highpass_spatial_filter(rec1)

    array_names = ("temporal_bins", "spatial_bins", "motion")
    motion_info = {}
    for name in array_names:
        if (motion_folder / f"{name}.npy").exists():
            motion_info[name] = np.load(motion_folder / f"{name}.npy")
        else:
            motion_info[name] = None

    if motion_info['motion'] is None:
        rec_corrected = rec1
    else:
        rec_corrected = interpolate_motion(
            recording=rec1,
            motion=motion_info['motion'],
            temporal_bins=motion_info['temporal_bins'],
            spatial_bins=motion_info['spatial_bins'])

    params_kilosort2 = si.get_default_sorter_params('kilosort2')
    params_kilosort2['skip_kilosort_preprocessing'] = False
    params_kilosort2['delete_recording_dat'] = True
    print(params_kilosort2)
    Kilosort2Sorter.set_kilosort2_path('sorting/Kilosort-2.0')
    sorting = si.run_sorter('kilosort2', rec_corrected, output_folder=str(sorting_folder),
                            verbose=True, remove_existing_folder=True, **params_kilosort2)

    we = si.extract_waveforms(rec_corrected, sorting, folder=str(waveform_folder),
                              sparse=True, max_spikes_per_unit=500, ms_before=1.5, ms_after=2.,
                              **job_kwargs)
