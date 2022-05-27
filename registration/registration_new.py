import numpy as np
import os
import glob
import spikeglx
import registration.estimate_displacement as ed
from pathlib import Path
from tqdm.auto import trange
from neurodsp import voltage, utils
import shutil


def registration(config):
    # This implementation has been tested with Neuropixels 1.0
    geomarray = config['script_dir'] + 'registration/spikes_localization_registration/channels_maps/np1_channel_map.npy'

    # I've only tested the spikeglx data reader that's part of ibllib (pip install ibllib)
    # yass is the default reader, but I've removed any mandatory yass imports in case you don't have that
    reader_type = 'spikeglx'
    # We only have to detect spikes once per dataset, then we can run the registration multiple times to test parameters
    detect_spikes = config['Registration']['detect_spikes']
    # I've found non-rigid registration to be optimal, but it can introduce artifacts for some datasets
    reg_win_num = config['Registration']['reg_win_num']
    reg_block_num = config['Registration']['reg_block_num']
    registration_type = config['Registration']['registration_type']
    horz_smooth = config['Registration']['horz_smooth']

    folders = glob.glob(config['neuropixel'] + '/*_g*')
    for pixel in range(config['num_neuropixels']):
        working_directory = folders[pixel] + '/'
        registration_directory = working_directory + 'NeuropixelsRegistration2/'
        if not os.path.exists(registration_directory):
            os.makedirs(registration_directory)

        # Prepare the data loader
        file = glob.glob(working_directory + '*_t*.imec' + str(pixel) + '.ap.bin')
        if len(file) != 1:
            raise SystemExit('Invalid Neuropixel data: ' + str(file))
        binary = Path(file[0])

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
            nbatches = int(np.ceil((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5))
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


        os.system('python ' + config['script_dir'] +
                  '/registration/spikes_localization_registration/scripts/subtract.py '
                  + str(standardized_file) + ' ' + registration_directory +
                  ' --noresidual --nowaveforms --dndetect --thresholds=12,10,8,6 --n_jobs=1 --geom=' +
                  config['script_dir'] +
                  '/registration/spikes_localization_registration/channels_maps/np1_channel_map.npy')

        import h5py
        import matplotlib.pyplot as plt
        from registration.spikes_localization_registration.subtraction_pipeline.ibme import fast_raster

        registered_file = glob.glob(registration_directory + 'subtraction_*.h5')
        with h5py.File(registered_file[0], "r") as f:
            x = f["localizations"][:, 0]
            z_orig = f["localizations"][:, 2]
            z_reg = f["z_reg"][:]
            time = f["spike_index"][:, 0] / 30_000
            maxptp = f["maxptps"][:]

        r_orig, *_ = fast_raster(maxptp, z_orig, time)
        r_reg, *_ = fast_raster(maxptp, z_reg, time)

        fig, (aa, ab) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=200)

        aa.imshow(np.clip(r_orig, 3, 13), aspect=0.5 * r_orig.shape[1] / r_orig.shape[0], cmap=plt.cm.inferno)
        ab.imshow(np.clip(r_reg, 3, 13), aspect=0.5 * r_reg.shape[1] / r_reg.shape[0], cmap=plt.cm.inferno)

        aa.set_ylabel("original depth")
        ab.set_ylabel("registered depth")
        ab.set_xlabel("time (s)")

        plt.savefig(registration_directory + 'raster.png')


        # create a new binary file with the drift corrected data ('standardized.bin')
        # this file does not contain the digital sync channel, so use your original file for that
        ed.register(reader, geomarray, total_shift, reader_type=reader_type,
                    registration_type=registration_type,
                    working_directory=registration_directory)