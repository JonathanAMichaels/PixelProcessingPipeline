import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import torch
from torch import nn


pretrained_path = (
    Path(__file__).parent.parent / "pretrained_denoiser/denoise.pt"
)


class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(feat2, feat3, size3), nn.ReLU())
        n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model=pretrained_path):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self


def enforce_decrease(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp.argmax()

    max_chan_even = max_chan - max_chan % 2
    max_chan_odd = max_chan_even + 1

    len_reg = (max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_even - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(0, max_chan_even - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_even + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_even + 4, n_chan, 2)] /= regularizer

    len_reg = (max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_odd - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(1, max_chan_odd - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_odd + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_odd + 4, n_chan, 2)] /= regularizer

    return wf


def enforce_decrease_np1(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp[16:28].argmax() + 16

    max_chan_a = max_chan - max_chan % 4
    for i in range(4, max_chan_a, 4):
        if wf[:, max_chan_a - i - 4].ptp() > wf[:, max_chan_a - i].ptp():
            wf[:, max_chan_a - i - 4] = (
                wf[:, max_chan_a - i - 4]
                * wf[:, max_chan_a - i].ptp()
                / wf[:, max_chan_a - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_a - 4, 4):
        if wf[:, max_chan_a + i + 4].ptp() > wf[:, max_chan_a + i].ptp():
            wf[:, max_chan_a + i + 4] = (
                wf[:, max_chan_a + i + 4]
                * wf[:, max_chan_a + i].ptp()
                / wf[:, max_chan_a + i + 4].ptp()
            )

    max_chan_b = max_chan - max_chan % 4 + 1
    for i in range(4, max_chan_b, 4):
        if wf[:, max_chan_b - i - 4].ptp() > wf[:, max_chan_b - i].ptp():
            wf[:, max_chan_b - i - 4] = (
                wf[:, max_chan_b - i - 4]
                * wf[:, max_chan_b - i].ptp()
                / wf[:, max_chan_b - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_b - 3, 4):
        if wf[:, max_chan_b + i + 4].ptp() > wf[:, max_chan_b + i].ptp():
            wf[:, max_chan_b + i + 4] = (
                wf[:, max_chan_b + i + 4]
                * wf[:, max_chan_b + i].ptp()
                / wf[:, max_chan_b + i + 4].ptp()
            )

    max_chan_c = max_chan - max_chan % 4 + 2
    for i in range(4, max_chan_c, 4):
        if wf[:, max_chan_c - i - 4].ptp() > wf[:, max_chan_c - i].ptp():
            wf[:, max_chan_c - i - 4] = (
                wf[:, max_chan_c - i - 4]
                * wf[:, max_chan_c - i].ptp()
                / wf[:, max_chan_c - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_c - 2, 4):
        if wf[:, max_chan_c + i + 4].ptp() > wf[:, max_chan_c + i].ptp():
            wf[:, max_chan_c + i + 4] = (
                wf[:, max_chan_c + i + 4]
                * wf[:, max_chan_c + i].ptp()
                / wf[:, max_chan_c + i + 4].ptp()
            )

    max_chan_d = max_chan - max_chan % 4 + 3
    for i in range(4, max_chan_d, 4):
        if wf[:, max_chan_d - i - 4].ptp() > wf[:, max_chan_d - i].ptp():
            wf[:, max_chan_d - i - 4] = (
                wf[:, max_chan_d - i - 4]
                * wf[:, max_chan_d - i].ptp()
                / wf[:, max_chan_d - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_d - 3, 4):
        if wf[:, max_chan_d + i + 4].ptp() > wf[:, max_chan_d + i].ptp():
            wf[:, max_chan_d + i + 4] = (
                wf[:, max_chan_d + i + 4]
                * wf[:, max_chan_d + i].ptp()
                / wf[:, max_chan_d + i + 4].ptp()
            )

    return wf


def make_shell(channel, geom, n_jumps=1):
    """See make_shells"""
    pt = geom[channel]
    dists = cdist([pt], geom).ravel()
    radius = np.unique(dists)[1 : n_jumps + 1][-1]
    return np.setdiff1d(np.flatnonzero(dists <= radius + 1e-8), [channel])


def make_shells(geom, n_jumps=1):
    """Get the neighbors of a channel within a radius

    That radius is found by figuring out the distance to the closest channel,
    then the channel which is the next closest (but farther than the closest),
    etc... for n_jumps.

    So, if n_jumps is 1, it will return the indices of channels which are
    as close as the closest channel. If n_jumps is 2, it will include those
    and also the indices of the next-closest channels. And so on...

    Returns
    -------
    shell_neighbors : list
        List of length geom.shape[0] (aka, the number of channels)
        The ith entry in the list is an array with the indices of the neighbors
        of the ith channel.
        i is not included in these arrays (a channel is not in its own shell).
    """
    return [make_shell(c, geom, n_jumps=n_jumps) for c in range(geom.shape[0])]


def make_radial_order_parents(
    geom, channel_index, n_jumps_per_growth=1, n_jumps_parent=2
):
    """Pre-computes a helper data structure for enforce_decrease_shells

    channel_index should be distance ordered
    """
    n_channels = len(channel_index)

    # which channels should we consider as possible parents for each channel?
    shells = make_shells(geom, n_jumps=n_jumps_parent)

    radial_parents = []
    for channel, neighbors in enumerate(channel_index):
        channel_parents = []

        # the closest shell will do nothing
        already_seen = [channel]
        shell0 = make_shell(channel, geom, n_jumps=n_jumps_per_growth)
        already_seen += sorted(c for c in shell0 if c not in already_seen)

        # so we start at the second jump
        jumps = 2
        while len(already_seen) < (neighbors < n_channels).sum():
            # grow our search -- what are the next-closest channels?
            new_shell = make_shell(
                channel, geom, n_jumps=jumps * n_jumps_per_growth
            )
            new_shell = list(
                sorted(
                    c
                    for c in new_shell
                    if (c not in already_seen) and (c in neighbors)
                )
            )

            # for each new channel, find the intersection of the channels
            # from previous shells and that channel's shell in `shells`
            for new_chan in new_shell:
                parents = np.intersect1d(shells[new_chan], already_seen)
                parents_rel = np.flatnonzero(np.isin(neighbors, parents))
                if not len(parents_rel):
                    # this can happen for some strange geometries
                    # in that case, let's just bail.
                    continue
                channel_parents.append(
                    (np.flatnonzero(neighbors == new_chan).item(), parents_rel)
                )

            # add this shell to what we have seen
            already_seen += new_shell
            jumps += 1

        radial_parents.append(channel_parents)

    return radial_parents


def enforce_decrease_shells(
    waveforms, maxchans, radial_parents, in_place=False
):
    """Radial enforce decrease"""
    N, T, C = waveforms.shape
    assert maxchans.shape == (N,)

    # compute original ptps and allocate storage for decreasing ones
    orig_ptps = waveforms.ptp(1)
    decreasing_ptps = orig_ptps.copy()

    # loop to enforce ptp decrease
    for i in range(N):
        decr_ptp = decreasing_ptps[i]
        for c, parents_rel in radial_parents[maxchans[i]]:
            if decr_ptp[c] > decr_ptp[parents_rel].max():
                decr_ptp[c] *= decr_ptp[parents_rel].max() / decr_ptp[c]

    # apply decreasing ptps to the original waveforms
    return np.multiply(
        waveforms,
        (decreasing_ptps / orig_ptps)[:, None, :],
        out=waveforms if in_place else None,
    )
