"""
This module if for basic util functions for visualization and quality control of the spike sorting.
It also includes wrappers to perform detections and denoising on waveforms
"""

"""
Detects and Denoise spikes within a numpy array of voltage traces
The goal is to benchmark the spike detector
"""
from pathlib import Path

import numpy as np
import h5py
import scipy
import torch
import tqdm
import shutil
import tqdm

from neurodsp import voltage
import spikeglx

from detect.run import find_channel_neighbors, make_channel_index
from detect.deduplication import deduplicate_gpu
import detect.detector
from localization_pipeline.denoiser import Denoise
from subtraction_pipeline import subtract, ibme


def run_cbin_ibl(cbin_file, standardized_file, t_start=0, t_end=None, **kwargs):
    """
    Pipelines:
    -   destriping
    -   subtraction / localisation
    -   registration
    Prototype for reproducible datasets to run on parede servers
    """
    cbin_file = Path(cbin_file)
    standardized_file = Path(standardized_file)
    standardized_dir = standardized_file.parent
    sr = spikeglx.Reader(cbin_file)
    h = sr.geometry
    if not standardized_file.exists():
        standardized_dir.mkdir(exist_ok=True, parents=True)
        batch_size_secs = 1
        batch_intervals_secs = 50
        # scans the file at constant interval, with a demi batch starting offset
        nbatches = int(np.floor((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5))
        wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
        for ibatch in tqdm.trange(nbatches, desc="destripe batches"):
            ifirst = int((ibatch + 0.5) * batch_intervals_secs * sr.fs + batch_intervals_secs)
            ilast = ifirst + int(batch_size_secs * sr.fs)
            sample = voltage.destripe(sr[ifirst:ilast, :-sr.nsync].T, fs=sr.fs, neuropixel_version=1)
            np.fill_diagonal(wrots[ibatch, :, :], 1 / voltage.rms(sample) * sr.sample2volts[:-sr.nsync] )

        wrot = np.median(wrots, axis=0)
        voltage.decompress_destripe_cbin(
            sr.file_bin, h=h, wrot=wrot, output_file=standardized_file, dtype=np.float32, nc_out=sr.nc - sr.nsync)
        # also copy the companion meta-data file
        shutil.copy(sr.file_meta_data, standardized_file.parent.joinpath(f"{sr.file_meta_data.stem}.normalized.meta"))
    sub_h5 = standardized_dir.joinpath(f"subtraction_{standardized_file.stem}_t_{t_start}_{t_end}.h5")
    if sub_h5.exists():
        return sub_h5
    sub_h5 = subtract.subtraction(standardized_file, standardized_file.parent, t_start=t_start, t_end=t_end, **kwargs)

    # -- registration
    with h5py.File(sub_h5, "r+") as h5:
        samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
        z_abs = h5["localizations"][:, 2]
        maxptps = h5["maxptps"]

        z_reg, dispmap = ibme.register_nonrigid(
            maxptps,
            z_abs,
            samples / sr.fs,
            robust_sigma=1,
            rigid_disp=200,
            disp=100,
            denoise_sigma=0.1,
            n_windows=10,
            widthmul=0.5,
        )
        z_reg -= (z_reg - z_abs).mean()
        dispmap -= dispmap.mean()

        h5.create_dataset("z_reg", data=z_reg)
        h5.create_dataset("dispmap", data=dispmap)
    return sub_h5


def h5_to_npy(h5_file, output_dir):
    h5 = h5py.File(h5_file, "r")
    to_keep = ['channel_index', 'dispmap', 'end_sample', 'first_channels', 'geom', 'localizations', 'maxptps',
               'spike_index', 'start_sample', 'tpca_components', 'tpca_mean', 'z_reg']
    output_dir.mkdir(exist_ok=True)
    for k in h5.keys():
        if k not in to_keep:
            continue
        np.save(output_dir.joinpath(f"{k}.npy"), h5[k])


def load_npy_yasap(npy_dir, fs=30000, registration=False):
    channels = {}
    channels['lateral_um'], channels['axial_um'] = np.hsplit(np.load(npy_dir.joinpath("geom.npy")), 2)
    spikes = {}
    spikes['amps'] = np.load(npy_dir.joinpath("maxptps.npy"))
    spikes['samples'] = np.load(npy_dir.joinpath("spike_index.npy"))[:, 0]
    spikes['raw_channels'] = np.load(npy_dir.joinpath("spike_index.npy"))[:, 1]
    spikes['times'] = spikes['samples'] / fs
    if registration:
        spikes['depths'] = np.load(npy_dir.joinpath("z_reg.npy"))
    else:
        loc = np.load(npy_dir.joinpath("localizations.npy"))
        spikes['depths'] = loc[:, 2]

    return spikes, channels


def plot_spikes_view_ephys(spikes, clusters, t0, nsecs, fs, rgb, label, eqcs):
    """
    Add spikes to a raw data viewer
    """
    stimes = spikes['samples'] / fs
    # stimes = spikes['times']
    slice_spikes = slice(np.searchsorted(stimes, t0), np.searchsorted(stimes, t0 + nsecs))
    t = (stimes[slice_spikes] - t0) * 1e3
    c = clusters.channels[spikes.clusters[slice_spikes]]
    for k in eqcs:
        eqcs[k].ctrl.add_scatter(t, c, rgb, label=label)


def init_detector(cbin_file):
    sr = spikeglx.Reader(cbin_file)
    REPO_PATH = Path(detect.detector.__file__).parents[1]
    APPLY_NN = True
    BATCH_SIZE_SECS = 1
    DETECT_THRESHOLD = -4  # in normalized units for once
    params = dict(
        apply_nn=APPLY_NN,  # If set to false, run voltage threshold instead of NN detector,
        detect_threshold=.56 if APPLY_NN else 6,  # 0.5 if apply NN, 4/5/6 otherwise,
        filter_sizes_denoise=[5, 11, 21],
        geom_array=np.c_[sr.geometry['x'], sr.geometry['y']],
        len_recording=sr.rl, n_batches=sr.rl / 2,
        n_filters_denoise=[16, 8, 4],
        n_filters_detect=[16, 8, 8],
        n_processors=4,
        n_sec_chunk=BATCH_SIZE_SECS,
        n_sec_chunk_gpu_detect=.1,
        output_directory=cbin_file.parent.joinpath("detection"),
        path_nn_denoiser=REPO_PATH.joinpath('pretrained_denoiser/denoise.pt'),
        path_nn_detector=REPO_PATH.joinpath('pretrained_detector/detect_np1.pt'),
        run_chunk_sec='full',
        sampling_rate=sr.fs,
        spatial_radius=70,
        spike_size_nn=121,
        standardized_dtype='float32',
        standardized_path=None,
    )

    neigh_channels = find_channel_neighbors(params['geom_array'], params['spatial_radius'])
    channel_index = make_channel_index(neigh_channels, params['geom_array'])

    # need to run by small batches of 10000 samples
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = detect.detector.Detect(params['n_filters_detect'], params['spike_size_nn'], channel_index)
    detector.load(params['path_nn_detector'])
    detector.to(device)

    denoiser = Denoise(params['n_filters_denoise'], params['filter_sizes_denoise'], params['spike_size_nn'])
    denoiser.load(params['path_nn_denoiser'])
    denoiser.to(device)


    return detector, denoiser, params, channel_index


def detections_f1_score(ta, xya, tb, xyb, t_thresh=20 / 30000, d_thresh=50):
    """
    Computes metrics to compare spike sorting detections from different runs / algorithms
    ta / tb: np.array of spikes times
    xya / xyb: np.array of coordinates (complex if 2d coordinates)
    returns: a dictionary with metrics as keys
    """
    amatch = np.zeros(ta.size)
    bmatch = np.zeros(tb.size)
    for i in tqdm.tqdm(np.arange(ta.size)):
        first, last = np.searchsorted(tb, ta[i] + np.array([-1, 1]) * t_thresh)
        selb = np.arange(first, last)[np.abs(xya[i] - xyb[first:last]) <= d_thresh]

        if selb.size:
            amatch[i] = selb[0]
            bmatch[selb] = i
    fn = np.sum(amatch == 0)  # false negatives
    tp = np.sum(bmatch != 0)  # true positives
    fp = bmatch.size - tp  # false positives
    precision = tp / (tp + fp)  # how many retrieved items are relevant
    recall = tp / (tp + fn)  # how many relevant items are retrieved
    f1 = tp / (tp + (fp + fn) / 2)
    est = dict(fn=int(fn), tp=int(tp), fp=int(fp), precision=float(precision),
               recall=float(recall), f1=float(f1), n=int(bmatch.size))
    return est


def detect_nn(data, detector, denoiser, channel_index, params):
    """Apply the detector to a numpy array [nsamples, ntraces]"""
    # TODO: we need overlaps here
    DETECT_BATCH = 10000  # need to run by small size data in order not to run out of GPU memory
    nbatches = data.shape[0] / DETECT_BATCH
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert nbatches % 1 == 0
    all_detects = []
    for m in np.arange(nbatches):
        first = int(m * DETECT_BATCH)
        last = int((m + 1) * DETECT_BATCH)
        data_ = torch.FloatTensor(data[first: last, :]).to(device)
        spike_index, wfs = detector.get_spike_times(data_, threshold=params['detect_threshold'])

        wfs_denoised = denoiser(wfs)[0].data
        energy = (torch.max(wfs_denoised, 1)[0] - torch.min(wfs_denoised, 1)[0])

        # deduplicate
        spike_index_dedup = deduplicate_gpu(spike_index, energy, data_.shape, channel_index)

        detects = spike_index_dedup.detach().cpu().numpy()
        detects[:, 0] = detects[:, 0] + m * DETECT_BATCH
        all_detects.append(detects)

        del data_
        del wfs
        del wfs_denoised
        del energy
        del spike_index
        del spike_index_dedup

        torch.cuda.empty_cache()

    all_detects = np.concatenate(all_detects)

    # plt.plot(mmap[1:10000, 55])
    npz_batches = np.load(
        '/datadisk/Data/spike_sorting/benchmark/raw/8ca1a850-26ef-42be-8b28-c2e2d12f06d6/detection/batch/detect_00000.npz')
    # npz_batches.files ['spike_index', 'spike_index_dedup', 'minibatch_loc']
    spikes_thresh = npz_batches['spike_index_dedup'][0]

    return all_detects




def apply_ks2_whitening(raw, kwm, sr, channels):
    if 'rawInd' not in channels:
        _, iraw, _ = np.intersect1d(sr.geometry['x'] * 1e4 + sr.geometry['y'],
                       channels['lateral_um'] * 1e4 + channels['axial_um'], return_indices=True)
    else:
        iraw = channels['rawInd']
    iraw = np.sort(iraw)
    carbutt  = raw - np.mean(raw, axis=0)
    butter_kwargs = {'N': 3, 'Wn': np.array([300, 8000]) / sr.fs * 2, 'btype': 'bandpass'}
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    carbutt = scipy.signal.sosfiltfilt(sos, carbutt)
    ks2 = np.zeros_like(raw)
    ks2[iraw, :] = np.matmul(kwm, carbutt[iraw, :])
    scaling = np.std(carbutt)  # choose and apply a constant scaling throughout
    ks2 = ks2 * np.std(carbutt) / np.std(ks2)
    return ks2
