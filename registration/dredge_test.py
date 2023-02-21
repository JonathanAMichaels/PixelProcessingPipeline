import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from pathlib import Path
from tqdm.auto import tqdm, trange

from dredge.python.reglib import lfpreg, ap_filter

plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

data_home = Path("/cifs/pruszynski/Malfoy/020223/020223_g0/020223_g0_imec0").expanduser()
raw_dir = data_home
raw_lfp_bin = next(raw_dir.glob("*lf.bin"))
raw_lfp_bin

ppx_dir = data_home / "ppx"
ppx_dir.mkdir(exist_ok=True)
ppx_lfp_bin = ppx_dir / raw_lfp_bin.name
ppx_lfp_bin
#%%
# load geometry array in the usual n_channels x 2 format
geom = loadmat("../geometries/neuropixPhase3B1_kilosortChanMap.mat")
geom = np.c_[geom["xcoords"], geom["ycoords"]]
geom.shape

# this preprocessing runs a destriping routine from ibl-neuropixel,
# which is essentially standardization + CMR, resamples to 250Hz,
# and averages channels at the same depth.
# we also add a BP filter at the beginning since the hardware filter
# has a kind of slow roll off and some spikes get through.
# alternatively, you could set avg_depth=False and csd=True to compute
# the csd (which is done columnwise and then averaged across depth,
# which is smarter.)

if False:
    ap_filter.run_preprocessing(
        raw_lfp_bin,
        ppx_lfp_bin,
        geom=geom,
        fs=2500,
        bp=(0.5, 250),
        extra_channels=1,
        resample_to=250,
        lfp_destripe=True,
        avg_depth=False,
        csd=True,
    )

y_unique = np.unique(geom[:, 1])

# plot original
lfp0 = np.memmap(raw_lfp_bin, dtype=np.int16).reshape(-1, 385)[:, :-1]
plt.imshow(lfp0[250*2500:250*2500 + 10 * 2500].T, aspect=10, cmap=plt.cm.bone);
plt.title("raw lfp")
plt.axis("off")
plt.savefig('/home/ROBARTS/jmichaels/PixelProcessingPipeline/registration/plots/orig.png')

lfp = np.memmap(ppx_lfp_bin, dtype=np.float32).reshape(-1, y_unique.size)

# load 20 seconds starting at 250s
chunk = lfp[100 * 250 : 120 * 250]

# vis to check preprocessing went alright
fig, ax = plt.subplots()
ax.imshow(chunk.T, aspect=0.5 * np.divide(*chunk.shape), cmap=plt.cm.bone)
ax.set_yticks(
    np.arange(0, y_unique.size, 25),
    [f"{u} / {i}" for i, u in zip(np.arange(0, y_unique.size, 25), y_unique[np.arange(0, y_unique.size, 25)])],
)
ax.set_xticks(np.arange(0, 20 * 250, 5*250), np.arange(0, 20, 5))
ax.set_ylabel("depth (um) / unique depth channel")
ax.set_xlabel("time (s)");
plt.savefig('/home/ROBARTS/jmichaels/PixelProcessingPipeline/registration/plots/csd.png')

# recall that `lfp` is the full recording in a memmap (not in memory)
# this took about ~15mins on my laptop (no GPU) but is much faster on GPU
p = lfpreg.online_register_rigid(
    lfp.T,
    adaptive_mincorr_percentile=20)
#)



import scipy.io
scipy.io.savemat('/home/ROBARTS/jmichaels/PixelProcessingPipeline/registration/p.mat', {'p': p})

# plot the estimated displacement over the signal throughout the
# whole recording so we can see how it looks
plot_chunk_len_s = 10
between_chunks_s = 1000
fs = 250
chunk_starts_samples = np.arange(
    0, lfp.shape[0] - plot_chunk_len_s * fs, between_chunks_s * fs
)

fig, axes = plt.subplots(
    nrows=len(chunk_starts_samples),
    figsize=(5, 2 * len(chunk_starts_samples)),
)

for start, ax in zip(chunk_starts_samples, axes):
    end = start + plot_chunk_len_s * fs
    lfp_chunk = lfp[start:end]
    p_chunk = p[start:end]

    ax.imshow(lfp_chunk.T, aspect=0.5 * np.divide(*lfp_chunk.shape), cmap=plt.cm.bone)
    ax.plot(y_unique.size / 2 + p_chunk, color="r")
    ax.set_yticks(np.arange(0, y_unique.size, 25), y_unique[np.arange(0, y_unique.size, 25)])
    ax.set_xticks(
        np.arange(0, plot_chunk_len_s * fs, 2.5*fs),
        np.arange(start / fs, start / fs + plot_chunk_len_s, 2.5)
    )
    ax.set_ylabel("depth (um)")
    plt.savefig('/home/ROBARTS/jmichaels/PixelProcessingPipeline/registration/plots/test' + str(start) + '.png')

ax.set_xlabel("time (s)")
fig.tight_layout()

