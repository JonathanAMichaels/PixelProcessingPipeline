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
        chans = range(chan_list[myomatrix][0] - 1, chan_list[myomatrix][1] - 1)
        ts = len(session.recordnodes[0].recordings[0].continuous[0].timestamps)
        segs = np.round(np.linspace(0, ts, num=100, endpoint=True)).astype('int')
        bin_file = directory + '/data.bin'
        if not os.path.isfile(bin_file):
            with open(bin_file, 'wb') as f:
                # segment time into managable chunks
                # for each set
                for i in range(len(segs)-1):
                    trange = range(segs[i], segs[i+1])
                    data = session.recordnodes[0].recordings[0].continuous[0].samples[np.ix_(trange, chans)]
                    f.write(np.int16(data))
            f.close()
        sync_data = dict([])
        sync_data['sync'] = \
            np.array(session.recordnodes[0].recordings[0].continuous[0].samples[:, sync_chan]).astype('int')
        scipy.io.savemat(directory + '/sync.mat', sync_data, do_compression=True)

        params = {'perform_drift_registration': False}
        data_path = Path(bin_file)
        dir_path = Path(directory + '/sorted')  # by default uses the same folder as the dataset
        output_dir = dir_path
        add_default_handler(level='INFO')  # print output as the algorithm runs
        run(data_path, dir_path=dir_path, output_dir=output_dir,
            probe=myomatrix_bipolar_probe(), low_memory=False, **params)

        # correct params.py to point to the shifted data
        with open(str(output_dir) + '/params.py', 'w') as f:
            f.write("dat_path = 'proc.dat'\nn_channels_dat = " + len(chans) +
                    "\ndtype = 'int16'\noffset = 0\n" +
                    "hp_filtered = True\nsample_rate = 30000\ntemplate_scaling = 20.0")

