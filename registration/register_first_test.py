from spikeglx import Reader
import estimate_displacement as ed
from utils import mat2npy
import glob
import h5py

# This implementation has been tested with Neuropixels 1.0
geomarray = mat2npy('~/PixelProcessingPipeline/geometries/neuropixPhase3B1_kilosortChanMap.mat') # convert .mat chan file to .npy chan file
reader = Reader('/cifs/pruszynski/Malfoy/011322/011322_g0/011322_g0_imec0/011322_g0_t0.imec0.ap.bin')

registration_directory = '/cifs/pruszynski/Malfoy/011322/011322_g0/011322_g0_imec0/NeuropixelsRegistration2/'
registered_file = glob.glob(registration_directory + 'subtraction_*.h5')
with h5py.File(registered_file[0], "r") as f:
    x = f["localizations"][:, 0]
    z_orig = f["localizations"][:, 2]
    z_reg = f["z_reg"][:]
    time = f["spike_index"][:, 0] / 30_000
    maxptp = f["maxptps"][:]
    dispmap = f["dispmap"][:]

# create a new binary file with the drift corrected data ('standardized.bin')
# this file does not contain the digital sync channel, so use your original file for that
ed.register(reader, geomarray, dispmap, reader_type='spikeglx',
            registration_type='non_rigid',
            working_directory=registration_directory)