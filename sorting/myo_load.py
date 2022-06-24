import glob
from pathlib import Path
from open_ephys.analysis import Session

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
        # for each set
        data = session.recordnodes[0].recordings[0].continuous[0].samples[:, chans]
        sync = session.recordnodes[0].recordings[0].continuous[0].samples[:, sync_chan]


