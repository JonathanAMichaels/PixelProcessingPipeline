import os
import fnmatch
import numpy as np
from sorting.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital
import spikeglx
import scipy.io
from scipy import signal
from scipy.io import savemat as savemat
import h5py
import glob
import spikeinterface.full as si
import shutil
from pathlib import Path


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


def extract_LFP_legacy(config_kilosort):
    data = spikeglx.Reader(Path(config_kilosort['neuropixel']))
    meta = data.geometry
    params = {'LFP_filter_type': 'scipy.signal.sosfiltfilt', 'bandpass_frequency': (0.5, 400), 'butterworth_order': 4,
              'sampling_rate': 1000, 'gain': 1e8}
    sos = signal.butter(params['butterworth_order'], params['bandpass_frequency'],
                        fs=int(data.fs), btype='bandpass', output='sos')
    all_data = np.zeros((int(data.ns/30), data.nc-1), dtype=np.int16)
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
        all_data[ind, :] = temp[0:len(ind), :] * params['gain']
    data.close()

    with open(config_kilosort['neuropixel_folder'] + '/LFP.npy', 'wb') as f:
        np.save(f, all_data)
    registered_file = glob.glob(config_kilosort['neuropixel_folder'] +
                                '/NeuropixelsRegistration2/' + 'subtraction_*.h5')
    with h5py.File(registered_file[0], "r") as f:
        disp_map = f["dispmap"][:]
    #with open(config_kilosort['neuropixel_folder'] + '/LFP_params.npy', 'wb') as f:
    #    np.save(f, {'sample_shift': meta['sample_shift'], 'electrode_x_um': meta['x'],
    #                'electrode_y_um': meta['y'], 'params': params, 'displacement_map': disp_map})

    savemat(config_kilosort['neuropixel_folder'] + '/LFP_params.mat',  {'sample_shift': meta['sample_shift'],
                                                                 'electrode_x_um': meta['x'],
                                                                 'electrode_y_um': meta['y'], 'params': params,
                                                                 'displacement_map': disp_map},
            do_compression=False)


def extract_LFP(config_kilosort):
    spikeglx_folder = Path(config_kilosort['neuropixel'])
    lfp_folder = spikeglx_folder / 'lfp'
    shutil.rmtree(lfp_folder)

    # global kwargs for parallel computing
    job_kwargs = dict(
        n_jobs=16,
        chunk_duration='1s',
        progress_bar=True,
    )

    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    print(stream_names)
    raw_rec = si.read_spikeglx(spikeglx_folder, stream_name=stream_names[0], load_sync_channel=False)
    rec_filtered = si.bandpass_filter(raw_rec, freq_min=1., freq_max=300.)
    rec_shifted = si.phase_shift(rec_filtered)
    rec_preprocessed = si.resample(rec_shifted, 1000)
    rec_preprocessed = si.scale(rec_preprocessed, rec_preprocessed.get_channel_gains())
    rec_preprocessed.save(folder=lfp_folder, dtype='int16', **job_kwargs)
