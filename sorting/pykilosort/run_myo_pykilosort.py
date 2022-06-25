import os
import sys

script_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_folder)

import glob
from pathlib import Path
from open_ephys.analysis import Session
import numpy as np
import scipy.io
import shutil
from pykilosort import run, add_default_handler, myomatrix_bipolar_probe, myomatrix_unipolar_probe


def myo_sort(config):
    directory = config['myomatrix']
    session = Session(directory)
    chan_list = config['Session']['myo_chan_list']
    sync_chan = int(config['Session']['myo_analog_chan']) - 1
    num_myomatrix = len(chan_list)
    for myomatrix in range(num_myomatrix):
        chans = range(chan_list[myomatrix][0] - 1, chan_list[myomatrix][1])
        ts = session.recordnodes[0].recordings[0].continuous[0].timestamps.shape[0]
        segs = np.round(np.linspace(0, ts, num=100, endpoint=True)).astype('int')
        bin_file = directory + '/data.bin'
        os.remove(bin_file)
        with open(bin_file, 'wb') as f:
            # segment time into manageable chunks
            # for each set
            for i in range(len(segs)-1):
                trange = range(segs[i], segs[i+1])
                data = session.recordnodes[0].recordings[0].continuous[0].samples[np.ix_(trange, chans)]
                f.write(np.int16(data))
        f.close()
        sync_data = dict([])
        sync_data['sync'] = session.recordnodes[0].recordings[0].continuous[0].samples[:, sync_chan]
        scipy.io.savemat(directory + '/sync.mat', sync_data, do_compression=True)

        params = {'perform_drift_registration': False, 'n_channels': len(chans)}
        data_path = Path(bin_file)
        dir_path = Path(directory + '/sorted')  # by default uses the same folder as the dataset
        output_dir = dir_path
        add_default_handler(level='INFO')  # print output as the algorithm runs
        if len(chans) == 16:
            probe = myomatrix_bipolar_probe()
        elif len(chans) == 32:
            probe = myomatrix_unipolar_probe()
        else:
            error('No probe configuration available')
        run(data_path, dir_path=dir_path, output_dir=output_dir,
            probe=probe, low_memory=False, **params)
        post_file = glob.glob(str(dir_path) + '/.kilosort/*/proc.dat')
        shutil.move(post_file[0], str(dir_path) + '/proc.dat')
        shutil.rmtree(dir_path.joinpath(".kilosort"), ignore_errors=True)

        # correct params.py to point to the shifted data
        with open(str(output_dir) + '/params.py', 'w') as f:
            f.write("dat_path = 'proc.dat'\nn_channels_dat = " + str(len(chans)) +
                    "\ndtype = 'int16'\noffset = 0\n" +
                    "hp_filtered = True\nsample_rate = 30000\ntemplate_scaling = 20.0")

        os.remove(bin_file)
