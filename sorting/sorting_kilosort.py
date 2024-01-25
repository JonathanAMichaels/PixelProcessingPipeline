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
import urllib.request
from pathlib import Path
from tqdm import tqdm
from kilosort import run_kilosort
from probeinterface import ProbeGroup
from probeinterface import write_prb, read_prb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import glob
import torch

torch.cuda.empty_cache()

def sorting(config):
    dataset_folder = Path(config['neuropixel_folder'])
    sorting_folder = dataset_folder / 'kilosort4'

    spikeglx_folder = dataset_folder
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    print(stream_names)
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_name=stream_names[0], load_sync_channel=False)

    P = raw_rec.get_probe()
    PRB = ProbeGroup()
    PRB.add_probe(P)
    write_prb(str(dataset_folder / 'probemap.prb'), PRB)

    SAVE_PATH = Path(glob.glob(str(dataset_folder) + "/*_t*.imec*.ap.bin")[0])

    # NOTE: 'n_chan_bin' is a required setting, and should reflect the total number
    #       of channels in the binary file. For information on other available
    #       settings, see `kilosort.run_kilosort.default_settings`.
    settings = {'data_dir': SAVE_PATH.parent, 'n_chan_bin': 385, 'nblocks': 3, 'batch_size': 60000, 'sig_interp': 60}

    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = \
        run_kilosort(settings=settings, probe_name=SAVE_PATH.parent / 'probemap.prb')


    # outputs saved to results_dir
    results_dir = sorting_folder
    ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
    camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
    contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
    chan_map = np.load(results_dir / 'channel_map.npy')
    templates = np.load(results_dir / 'templates.npy')
    chan_best = (templates ** 2).sum(axis=1).argmax(axis=-1)
    chan_best = chan_map[chan_best]
    amplitudes = np.load(results_dir / 'amplitudes.npy')
    st = np.load(results_dir / 'spike_times.npy')
    clu = np.load(results_dir / 'spike_clusters.npy')
    firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
    dshift = ops['dshift']


    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    gray = .5 * np.ones(3)

    fig = plt.figure(figsize=(10, 10), dpi=100)
    grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(np.arange(0, ops['Nbatches']) * 2, dshift);
    ax.set_xlabel('time (sec.)')
    ax.set_ylabel('drift (um)')

    ax = fig.add_subplot(grid[0, 1:])
    t0 = 0
    t1 = np.nonzero(st > ops['fs'] * 5)[0][0]
    ax.scatter(st[t0:t1] / 30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
    ax.set_xlim([0, 5])
    ax.set_ylim([chan_map.max(), 0])
    ax.set_xlabel('time (sec.)')
    ax.set_ylabel('channel')
    ax.set_title('spikes from units')

    ax = fig.add_subplot(grid[1, 0])
    nb = ax.hist(firing_rates, 20, color=gray)
    ax.set_xlabel('firing rate (Hz)')
    ax.set_ylabel('# of units')

    ax = fig.add_subplot(grid[1, 1])
    nb = ax.hist(camps, 20, color=gray)
    ax.set_xlabel('amplitude')
    ax.set_ylabel('# of units')

    ax = fig.add_subplot(grid[1, 2])
    nb = ax.hist(np.minimum(100, contam_pct), np.arange(0, 105, 5), color=gray)
    ax.plot([10, 10], [0, nb[0].max()], 'k--')
    ax.set_xlabel('% contamination')
    ax.set_ylabel('# of units')
    ax.set_title('< 10% = good units')

    for k in range(2):
        ax = fig.add_subplot(grid[2, k])
        is_ref = contam_pct < 10.
        ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
        ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_xlabel('firing rate (Hz)')
        ax.legend()
        if k == 1:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('loglog')

    plt.savefig(sorting_folder / 'results_fig.png')
