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

def registration(config):
    folders = glob.glob(config['neuropixel'] + '/*_g*')
    for pixel in range(config['num_neuropixels']):

        dataset_folder = Path(folders[pixel] + '/')
        motion_folder = dataset_folder / 'motion'
        peak_folder = dataset_folder / 'peaks'
        peak_folder.mkdir(exist_ok=True)

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

        P = raw_rec.get_probe()
        from probeinterface import ProbeGroup
        PRB = ProbeGroup()
        PRB.add_probe(P)
        #from probeinterface import write_prb, read_prb
        #write_prb('NPnew.prb', PRB)

        # preprocessing 1 : bandpass (this is smoother) + cmr
        rec1 = si.bandpass_filter(recording=raw_rec, freq_min=300., freq_max=5000.)
        rec1 = si.phase_shift(rec1)
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec1, noisy_channel_threshold=0.5,
                                                                 dead_channel_threshold=-0.1, chunk_duration_s=0.5,
                                                                 num_random_chunks=100)
        print(bad_channel_ids)
        rec1 = rec1.remove_channels(bad_channel_ids)
        # rec_bad = interpolate_bad_channels(rec_shifted, bad_channel_ids)
        rec1 = highpass_spatial_filter(rec1)

        # here the corrected recording is done on the preprocessing 1
        # rec_corrected1 will not be used for sorting!
        rec_corrected1 = correct_motion(recording=rec1, preset="nonrigid_accurate",
                                        detect_kwargs=dict(detect_threshold=6.), folder=motion_folder, **job_kwargs)

        # preprocessing 2 : highpass + cmr
        rec2 = si.bandpass_filter(recording=raw_rec, freq_min=300.)
        rec2 = si.phase_shift(rec2)
        bad_channel_ids, channel_labels = si.detect_bad_channels(rec2, noisy_channel_threshold=0.5,
                                                                 dead_channel_threshold=-0.1, chunk_duration_s=0.5,
                                                                 num_random_chunks=100)
        print(bad_channel_ids)
        rec2 = rec2.remove_channels(bad_channel_ids)
        # rec_bad = interpolate_bad_channels(rec_shifted, bad_channel_ids)
        rec2 = highpass_spatial_filter(rec2)
        # we use another preprocessing for the final interpolation

        motion_info = load_motion_info(motion_folder)
        rec_corrected = interpolate_motion(
            recording=rec2,
            motion=motion_info['motion'],
            temporal_bins=motion_info['temporal_bins'],
            spatial_bins=motion_info['spatial_bins'],
            **motion_info['parameters']['interpolate_motion_kwargs'])
        # and plot
        fig = plt.figure(figsize=(14, 8))
        si.plot_motion(motion_info, figure=fig, depth_lim=(400, 600),
                       color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
        # rec_corrected.save(folder=corrected_folder, format='binary', **job_kwargs)

        params_kilosort2_5 = si.get_default_sorter_params('kilosort2_5')
        params_kilosort2_5['do_correction'] = False
        params_kilosort2_5['skip_kilosort_preprocessing'] = False
        print(params_kilosort2_5)
        Kilosort2_5Sorter.set_kilosort2_5_path('Kilosort-2.5')
        sorting = si.run_sorter('kilosort2_5', rec_corrected, output_folder=dataset_folder / 'kilosort2.5_output',
                                verbose=True, **params_kilosort2_5)

        we = si.extract_waveforms(rec_corrected, sorting, folder=dataset_folder / 'waveforms_kilosort2.5',
                                  sparse=True, max_spikes_per_unit=500, ms_before=1.5, ms_after=2.,
                                  **job_kwargs)

        metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                               'isi_violation', 'amplitude_cutoff'])
        metrics