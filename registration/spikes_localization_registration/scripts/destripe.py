import argparse
import numpy as np
import shutil

from pathlib import Path
from tqdm.auto import trange

from neurodsp import voltage, utils
import spikeglx


ap = argparse.ArgumentParser()

ap.add_argument("binary")

args = ap.parse_args()


binary = Path(args.binary)

print("Destriping", binary)

folder = binary.parent
standardized_file = folder / f"{binary.stem}.normalized.bin"

# run destriping
sr = spikeglx.Reader(binary)
print(sr.nc, sr.nsync, sr.rl)
h = sr.geometry
if not standardized_file.exists():
    batch_size_secs = 1
    batch_intervals_secs = 50
    # scans the file at constant interval, with a demi batch starting offset
    nbatches = int(
        np.ceil((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5)
    )
    print(nbatches)
    wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
    for ibatch in trange(nbatches, desc="destripe batches"):
        ifirst = int(
            (ibatch + 0.5) * batch_intervals_secs * sr.fs
            + batch_intervals_secs
        )
        ilast = ifirst + int(batch_size_secs * sr.fs)
        sample = voltage.destripe(
            sr[ifirst:ilast, : -sr.nsync].T, fs=sr.fs, neuropixel_version=1
        )
        np.fill_diagonal(
            wrots[ibatch, :, :],
            1 / utils.rms(sample) * sr.sample2volts[: -sr.nsync],
        )

    wrot = np.median(wrots, axis=0)
    voltage.decompress_destripe_cbin(
        sr.file_bin,
        h=h,
        wrot=wrot,
        output_file=standardized_file,
        dtype=np.float32,
        nc_out=sr.nc - sr.nsync,
    )

    # also copy the companion meta-data file
    shutil.copy(
        sr.file_meta_data,
        standardized_file.parent.joinpath(
            f"{sr.file_meta_data.stem}.normalized.meta"
        ),
    )
