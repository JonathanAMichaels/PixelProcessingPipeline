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

channelRemap = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
                24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7]

def myo_sort(config):
    directory = config['myomatrix']
    #session = Session(directory)
    #chan_list = config['Session']['myo_chan_list']
    #sync_chan = int(config['Session']['myo_analog_chan']) - 1
    chans = range(0, config['num_chans'])
    #ts = session.recordnodes[0].recordings[0].continuous[0].timestamps.shape[0]
    #segs = np.round(np.linspace(0, ts, num=100, endpoint=True)).astype('int')
    bin_file = directory + '/data' + str(config['myomatrix_num']) + '.bin'
    #with open(bin_file, 'wb') as f:
        # segment time into manageable chunks
        # for each set
    #    for i in range(len(segs)-1):
    #        trange = range(segs[i], segs[i+1])
    #        data = session.recordnodes[0].recordings[0].continuous[0].samples[np.ix_(trange, chans)]
    #        if len(chans) == 32:
    #            data = data[:, channelRemap]
    #        f.write(np.int16(data))
    #f.close()
    #sync_data = dict([])
    #sync_data['sync'] = session.recordnodes[0].recordings[0].continuous[0].samples[:, sync_chan]
    #scipy.io.savemat(directory + '/sync.mat', sync_data, do_compression=True)

    params = {'perform_drift_registration': False, 'n_channels': len(chans), 'minfr_goodchannels': 0.1}
    data_path = Path(bin_file)
    dir_path = Path(config['myomatrix_folder'])  # by default uses the same folder as the dataset
    output_dir = dir_path
    add_default_handler(level='INFO')  # print output as the algorithm runs
    if len(chans) == 16:
        probe = myomatrix_bipolar_probe()
    elif len(chans) == 32:
        probe = myomatrix_unipolar_probe()
    else:
        error('No probe configuration available')
    run(data_path, dir_path=dir_path, output_dir=output_dir,
        probe=probe, low_memory=True, **params)
    post_file = glob.glob(str(dir_path) + '/.kilosort/*/proc.dat')
    try:
        shutil.move(post_file[0], str(dir_path) + '/proc.dat')
    except OSError:
        pass
    shutil.rmtree(dir_path.joinpath(".kilosort"), ignore_errors=True)

    # correct params.py to point to the shifted data
    with open(str(output_dir) + '/params.py', 'w') as f:
        f.write("dat_path = 'proc.dat'\nn_channels_dat = " + str(len(chans)) +
                "\ndtype = 'int16'\noffset = 0\n" +
                "hp_filtered = True\nsample_rate = 30000\ntemplate_scaling = 20.0")

    os.remove(bin_file)