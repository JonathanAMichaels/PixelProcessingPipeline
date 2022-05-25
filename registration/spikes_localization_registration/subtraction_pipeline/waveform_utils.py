import numpy as np
from scipy.spatial.distance import cdist


def channel_index_subset(geom, channel_index, n_channels=None, radius=None):
    """Restrict channel index to fewer channels

    Creates a boolean mask of the same shape as the channel index
    that you can use to restrict waveforms extracted using that
    channel index to fewer channels.

    Operates in two modes:
        n_channels is not None:
            This is the old style where we will grab the n_channels
            channels whose indices are nearest to the max chan index.
        radius is not None:
            Grab channels within a spatial radius.
    """
    subset = np.empty(shape=channel_index.shape, dtype=bool)
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=-2 * geom.max())
    for c in range(len(geom)):
        if n_channels is not None:
            low = max(0, c - n_channels // 2)
            low = min(len(geom) - n_channels, low)
            high = min(len(geom), low + n_channels)
            subset[c] = (low <= channel_index[c]) & (channel_index[c] < high)
        elif radius is not None:
            dists = cdist([geom[c]], pgeom[channel_index[c]]).ravel()
            subset[c] = dists <= radius
        else:
            subset[c] = True
    return subset


def relativize_waveforms_np1(
    wfs, firstchans_orig, geom, maxchans_orig, feat_chans=20
):
    """
    Extract fewer channels.
    """
    chans_down = feat_chans // 2
    chans_down -= chans_down % 4

    stdwfs = np.zeros(
        (wfs.shape[0], wfs.shape[1], feat_chans), dtype=wfs.dtype
    )

    firstchans_std = firstchans_orig.copy().astype(int)

    for i in range(wfs.shape[0]):
        wf = wfs[i]
        if maxchans_orig is None:
            mcrel = wf.ptp(0).argmax()
        else:
            mcrel = maxchans_orig[i] - firstchans_orig[i]
        mcrix = mcrel - mcrel % 4

        low, high = mcrix - chans_down, mcrix + feat_chans - chans_down
        if low < 0:
            low, high = 0, feat_chans
        if high > wfs.shape[2]:
            low, high = wfs.shape[2] - feat_chans, wfs.shape[2]

        firstchans_std[i] += low
        stdwfs[i] = wf[:, low:high]

    return stdwfs, firstchans_std, chans_down
