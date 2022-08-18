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
    print(meta)
    sos = signal.butter(4, (0.5, 300), fs=int(data.fs), btype='bandpass', output='sos')  # 300Hz lowpass filter
    all_data = np.zeros((int(data.ns/30), data.nc))
    for j in range(data.nc):
        start = time.time()
        print(j)
        temp = data._raw[:, j].astype(np.int16, copy=True)
        #temp = data.read(nsel=slice(0, data.ns), csel=j, sync=False)
        #temp = signal.sosfilt(sos, data.read(nsel=slice(0, data.ns), csel=j, sync=False))
        #temp = temp[::30]  # down-sample
        #all_data[:, j] = temp[0:all_data.shape[0]]
        end = time.time()
        print(end-start)
    data.close()
    save_data = dict([])
    save_data['LFP'] = all_data
    scipy.io.savemat(config_kilosort['neuropixel_folder'] + '/LFP.mat', save_data, do_compression=True)




