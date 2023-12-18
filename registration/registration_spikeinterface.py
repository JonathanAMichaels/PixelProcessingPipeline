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
import numpy as np
plt.rcParams["figure.figsize"] = (20, 12)

def registration(config):
    folders = glob.glob(config['neuropixel'] + '/*_g*')
    for pixel in range(config['num_neuropixels']):

        dataset_folder = Path(folders[pixel] + '/')
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
                                                                 num_random_chunks=10)
        print(bad_channel_ids)
        rec1 = rec1.remove_channels(bad_channel_ids)
        # rec_bad = interpolate_bad_channels(rec_shifted, bad_channel_ids)
        rec1 = highpass_spatial_filter(rec1)

        # Step 1 : activity profile
        #peaks = detect_peaks(recording=rec1, method="locally_exclusive", detect_threshold=8.0, **job_kwargs)
        #np.save(motion_folder / 'peaks.npy')
        peaks = np.load(motion_folder / 'peaks.npy')
        #peaks = select_peaks(peaks, method='smart_sampling_amplitudes', 1000000, **job_kwargs)
        #peak_locations = localize_peaks(recording=rec1, peaks=peaks, method="monopolar_triangulation", **job_kwargs)
        #np.save(motion_folder / 'peak_locations.npy', peak_locations)
        peak_locations = np.load(motion_folder / 'peak_locations.npy')

        # Step 2: motion inference
        motion, temporal_bins, spatial_bins = estimate_motion(recording=rec1,
                                                              peaks=peaks,
                                                              peak_locations=peak_locations,
                                                              method="decentralized",
                                                              win_step_um=100.0,
                                                              win_sigma_um=300.0,
                                                              post_clean=True,
                                                              progress_bar=True,
                                                              **{'corr_threshold': 0.6, 'conv_engine': 'torch'})
        np.save(motion_folder / "temporal_bins.npy", temporal_bins)
        np.save(motion_folder / "motion.npy", motion)
        if spatial_bins is not None:
            np.save(motion_folder / "spatial_bins.npy", spatial_bins)

        motion_info = load_motion_info(motion_folder)
        fig = plt.figure(figsize=(14, 8))
        si.plot_motion(motion_info, figure=fig,
                       color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
        plt.savefig(motion_folder / 'motion.png')
