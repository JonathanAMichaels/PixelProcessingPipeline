import os
import glob
from pathlib import Path
from open_ephys.analysis import Session
import numpy as np

def myo_load(config):
    directory = config['myomatrix']
    session = Session(directory)

    chan_list = config['Session']['myo_chan_list']
    sync_chan = int(config['Session']['myo_analog_chan']) - 1
    print(chan_list)
    print(sync_chan)
    num_myomatrix = len(chan_list)
    for myomatrix in range(num_myomatrix):
        chans = range(chan_list[myomatrix][0] - 1, chan_list[myomatrix][1] - 1)
        print(chans)
        fs = 30000
        ts = len(session.recordnodes[0].recordings[0].continuous[0].timestamps)
        segs = int(np.round(np.linspace(0, ts, num=100, endpoint=True)))
        bin_file = directory + '/data.bin'
        if not os.path.isfile(bin_file):
            with open(bin_file, 'wb') as f:
                # segment time into managable chunks
                # for each set
                for i in range(len(segs)-1):
                    trange = range(segs[i], segs[i+1])
                    data = session.recordnodes[0].recordings[0].continuous[0].samples[trange, chans]
                    f.write(np.int16(data))
            f.close()
            sync = session.recordnodes[0].recordings[0].continuous[0].samples[:, sync_chan]


