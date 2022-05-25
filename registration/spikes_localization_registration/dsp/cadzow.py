import numpy as np
import scipy.signal
from ibllib.ephys.neuropixel import trace_header
from ibllib.dsp import cadzow

h = trace_header(1)


def cadzow_np1(wav, fs=30000, rank=5, niter=1, fmax=7500):
    """
    Apply Fxy rank-denoiser to a full recording of Neuropixel 1 probe geometry
    :param wav: ntr, ns
    :param fs:
    :return:
    """
    # ntr - nswx has to be a multiple of (nswx - ovx)
    ntr, ns = wav.shape
    """
    try some window sizes:
     ovx is the overlap in x
     nswx is the size of the window in x
     npad is the padding
    """
    # ovx, nswx, npad = (int(5), int(33), int(6))
    ovx, nswx, npad = (int(16), int(32), int(0))
    # ovx, nswx, npad = (int(32), int(64), int(0))
    # ovx, nswx, npad = (int(24), int(64), int(0))
    # ovx, nswx, npad = (int(8), int(16), int(0))
    nwinx = int(np.ceil((ntr + npad * 2 - ovx) / (nswx - ovx)))
    fscale = scipy.fft.rfftfreq(ns, d=1 / fs)
    imax = np.searchsorted(fscale, fmax)
    WAV = scipy.fft.rfft(wav[:, :])
    padgain = scipy.signal.windows.hann(npad * 2)[:npad]
    WAV = np.r_[np.flipud(WAV[1:npad + 1, :]) * padgain[:, np.newaxis],
                WAV,
                np.flipud(WAV[-npad - 2: - 1, :]) * np.flipud(np.r_[padgain, 1])[:, np.newaxis]]  # apply padding
    x = np.r_[np.flipud(h['x'][1:npad + 1]), h['x'],  np.flipud(h['x'][-npad - 2: - 1])]
    y = np.r_[np.flipud(h['y'][1:npad + 1]) - 120, h['y'],  np.flipud(h['y'][-npad - 2: - 1]) + 120]
    WAV_ = np.zeros_like(WAV)
    gain = np.zeros(ntr + npad *2 + 1)
    hanning = scipy.signal.windows.hann(ovx * 2 - 1)[0:ovx]
    assert np.all(np.isclose(hanning + np.flipud(hanning), 1))
    gain_window = np.r_[hanning, np.ones(nswx - ovx * 2), np.flipud(hanning)]
    for firstx in np.arange(nwinx) * (nswx - ovx):
        lastx = int(firstx + nswx)
        if firstx == 0:
            gw = np.r_[hanning * 0 + 1, np.ones(nswx - ovx * 2) , np.flipud(hanning)]
        elif lastx == ntr:
            gw = np.r_[hanning, np.ones(nswx - ovx * 2), hanning * 0 + 1]
        else:
            gw = gain_window
        gain[firstx:lastx] += gw
        T, it, itr, trcount = cadzow.trajectory(x=x[firstx:lastx], y=y[firstx:lastx])
        array = WAV[firstx:lastx, :]
        print(firstx, lastx, x[firstx:lastx].shape, WAV[firstx:lastx, :].shape, T.shape)
        array = cadzow.denoise(array, x=x[firstx:lastx], y=y[firstx:lastx], r=rank, imax=imax, niter=niter)
        WAV_[firstx:lastx, :] += array * gw[:, np.newaxis]

    WAV_ = WAV_[npad:-npad - 1]  # remove padding
    wav_ = scipy.fft.irfft(WAV_)
    return wav_
