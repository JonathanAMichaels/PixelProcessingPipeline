import os
import glob
from spikeglx import Reader
import registration.estimate_displacement as ed
from registration.utils import mat2npy


def registration(config):
    # This implementation has been tested with Neuropixels 1.0
    geomarray = mat2npy(config['script_dir'] + '/geometries/neuropixPhase3B1_kilosortChanMap.mat') # convert .mat chan file to .npy chan file

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
        registration_directory = working_directory + 'NeuropixelsRegistration/'
        if not os.path.exists(registration_directory):
            os.makedirs(registration_directory)

        # Prepare the data loader
        file = glob.glob(working_directory + '*_t*.imec' + str(pixel) + '.ap.bin')
        if len(file) != 1:
            raise SystemExit('Invalid Neuropixel data: ' + str(file))
        reader = Reader(file[0])

        if detect_spikes:
            # detect spikes
            raster = ed.check_raster(reader, geomarray, reader_type=reader_type, num_chans_per_spike=4,
                                     detection_threshold=6,
                                     working_directory=registration_directory,
                                     save_raster_info=True)

        # run the registration
        # visualization will be saved to decentralized_raster/, shift will be saved as total_shift.npy
        total_shift = ed.estimate_displacement(reader, geomarray,
                                               reader_type=reader_type,
                                               num_chans_per_spike=4,
                                               detection_threshold=6,
                                               horz_smooth=horz_smooth,
                                               reg_win_num=reg_win_num,
                                               reg_block_num=reg_block_num,
                                               iteration_num=4,
                                               resume_with_raster=True,
                                               working_directory=registration_directory)


        # create a new binary file with the drift corrected data ('standardized.bin')
        # this file does not contain the digital sync channel, so use your original file for that
        ed.register(reader, geomarray, total_shift, reader_type=reader_type,
                    registration_type=registration_type,
                    working_directory=registration_directory)