import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from spikeinterface.preprocessing import highpass_spatial_filter
from spikeinterface.preprocessing import correct_motion
from spikeinterface.preprocessing.motion import load_motion_info
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
from spikeinterface.preprocessing.motion import load_motion_info
from spikeinterface.sorters import Kilosort2_5Sorter
plt.rcParams["figure.figsize"] = (20, 12)

def sorting(config):
    dataset_folder = Path(config['neuropixel_folder'])
    motion_folder = dataset_folder / 'motion'

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
    print(params_kilosort2_5)
    Kilosort2_5Sorter.set_kilosort2_5_path('sorting/Kilosort-2.5')
    sorting = si.run_sorter('kilosort2_5', rec_corrected, output_folder=dataset_folder / 'kilosort2.5_output',
                            verbose=True, **params_kilosort2_5)

    we = si.extract_waveforms(rec_corrected, sorting, folder=dataset_folder / 'waveforms_kilosort2.5',
                              sparse=True, max_spikes_per_unit=500, ms_before=1.5, ms_after=2.,
                              **job_kwargs)

    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                           'isi_violation', 'amplitude_cutoff'])
    metrics