import numpy as np
import os
import sys

script_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_folder)

import glob
from pathlib import Path
from open_ephys.analysis import Session

def myo_load(config):
    directory = config['myomatrix']
    session = Session(directory)

    chan_list = config['myo_chan_list']
    sync_chan = int(config['myo_analog_chan'])
    print(chan_list)
    # for each set
    data = session.recordingnodes[0].recordings[0].continuous[0].samples[:, 0]
    sync = session.recordingnodes[0].recordings[0].continuous[0].samples[:, sync_chan]


