import numpy as np
import scipy.signal
import seaborn as sns

from ibllib.atlas import BrainRegions
from neurodsp import voltage
from viewephys.gui import viewephys
from brainbox.io.spikeglx import stream

V_T0 = (60 * 10, 60 * 30, 60 * 50)  # raw data samples at 10, 30, 50 min in


def raw_data(pid, times=V_T0, channels=None, br=None, one=None, output_dir=None, ss=None):
    """
    Outputs 1 second views of the raw data, butterworth and destriped, with spike sorting overlayed
    """
    if channels is not None and br is None:
        br = BrainRegions()
    ss = ss or {}

    for t0 in times:
        sr, t0_bin = stream(pid, t0=t0, nsecs=1, one=one)
        destripe = voltage.destripe(sr[:, :-sr.nsync].T, fs=sr.fs, channel_labels=True)
        butter_kwargs = {'N': 3, 'Wn': 300 / sr.fs * 2, 'btype': 'highpass'}
        sos = scipy.signal.butter(**butter_kwargs, output='sos')
        butt = scipy.signal.sosfiltfilt(sos, sr[:, :-sr.nsync].T)
        evs = {}
        evs['butterworth'] = viewephys(data=butt, fs=sr.fs, channels=channels, br=br, title='butt')
        evs['destripe'] = viewephys(data=destripe, fs=sr.fs, channels=channels, br=br, title='ap_destripe')

        eqc_xrange = [620, 700]
        eqc_gain = - 90 + 120
        for k, ev in evs.items():
            ev.ctrl.set_gain(eqc_gain)
            ev.resize(1960, 1200)
            ev.viewBox_seismic.setXRange(*eqc_xrange)
            ev.viewBox_seismic.setYRange(0, sr.nc)
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True)
                ev.grab().save(str(output_dir.joinpath(f"{pid}_voltage_t{t0}_{k}.png")))

        for i, k in enumerate(ss):
            rgb = tuple(np.r_[(np.array(sns.color_palette('bright')[i]) * 255).astype(np.uint8), 150])
            spikes = ss[k]['spikes']
            istart, iend = np.searchsorted(spikes['times'], t0 + np.array([0, 1]))
            # here you can add some labels to color the spikes
            evs['destripe'].ctrl.add_scatter(
                x=(spikes['times'][istart:iend] - t0) * 1e3,
                y=spikes['raw_channels'][istart:iend],
                rgb=rgb
            )
            if output_dir is not None:
                ev.grab().save(str(output_dir.joinpath(f"{pid}_voltage_t{t0}_spikes_{k}.png")))
    return evs
