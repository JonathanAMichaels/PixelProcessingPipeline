import os
import fnmatch
import shutil
from pathlib import Path
import numpy as np
from sorting.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital
import spikeglx
import scipy.io
from scipy import signal
import time


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def create_config(script_folder, folder):
    shutil.copyfile(script_folder + '/config_template.yaml', folder + '/config.yaml')


def extract_sync(config_kilosort):
    meta = readMeta(Path(config_kilosort['neuropixel']))
    tStart = 0
    tEnd = np.floor(float(meta['fileTimeSecs']))
    sRate = SampRate(meta)
    firstSamp = int(sRate * tStart)
    lastSamp = int(sRate * tEnd)
    rawData = makeMemMapRaw(Path(config_kilosort['neuropixel']), meta)
    sync_data = dict([])
    sync_data['sync'] = ExtractDigital(rawData, firstSamp, lastSamp, 0, [6], meta)
    scipy.io.savemat(config_kilosort['neuropixel_folder'] + '/sync.mat', sync_data, do_compression=True)


def extract_LFP(config_kilosort):
    data = spikeglx.Reader(Path(config_kilosort['neuropixel']))
    meta = data.geometry
    params = {'LFP_filter_type': 'scipy.signal.sosfiltfilt', 'bandpass_frequency': (0.5, 400), 'butterworth_order': 4,
              'sampling_rate': 1000}
    sos = signal.butter(params['butterworth_order'], params['bandpass_frequency'],
                        fs=int(data.fs), btype='bandpass', output='sos')
    all_data = np.zeros((int(data.ns/30), data.nc-1), dtype=np.float32)
    all = list(range(data.ns))
    buffer_size = int(30000*4.2)
    intervals = all[0: int(data.ns) - buffer_size: 600000]
    I = np.zeros(2, dtype=np.int64)
    for i in range(len(intervals)):
        print('Time bin ' + str(i) + ' of ' + str(len(intervals)))
        if i == 0:
            I[0] = intervals[i]
        else:
            I[0] = intervals[i] - buffer_size
        if i == len(intervals)-1:
            I[1] = int(data.ns)
        else:
            I[1] = intervals[i+1] + buffer_size
        temp = signal.sosfilt(sos,
                              data.read(nsel=slice(I[0], I[1]), csel=slice(0, data.nc-1), sync=False),
                              axis=0)
        if i == 0:
            temp = temp[0:-buffer_size, :]
            ind = list(range(int(intervals[i] / 30), int(intervals[i + 1] / 30)))
        elif i == len(intervals)-1:
            temp = temp[buffer_size:, :]
            ind = list(range(int(intervals[i] / 30), int(data.ns / 30)))
        else:
            temp = temp[buffer_size: -buffer_size, :]
            ind = list(range(int(intervals[i] / 30), int(intervals[i + 1] / 30)))
        temp = temp[::30, :]  # down-sample
        all_data[ind, :] = temp[0:len(ind), :]
    data.close()

    with open(config_kilosort['neuropixel_folder'] + '/LFP.npy', 'wb') as f:
        np.save(f, all_data)
    with open(config_kilosort['neuropixel_folder'] + '/LFP_params.npy', 'wb') as f:
        np.save(f, {'sample_shift': meta['sample_shift'], 'electrode_x_um': meta['x'],
                    'electrode_y_um': meta['y'], 'params': params})




