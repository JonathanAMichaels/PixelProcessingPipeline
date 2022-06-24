import glob
from pathlib import Path
from open_ephys.analysis import Session

def myo_load(config):
    directory = config['myomatrix']
    session = Session(directory)

    chan_list = config['Session']['myo_chan_list']
    sync_chan = int(config['Session']['myo_analog_chan'])
    print(chan_list)
    # for each set
    data = session.recordingnodes[0].recordings[0].continuous[0].samples[:, 0]
    sync = session.recordingnodes[0].recordings[0].continuous[0].samples[:, sync_chan]


