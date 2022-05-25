import time
import os
import sys
import numpy as np
from numpy import linalg as LA
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from brainbox.singlecell import calculate_peths
from sklearn.decomposition import PCA

def read_waveforms(spike_times, geom_array, bin_file, dtype = 'float32', spike_size = 121, n_times=None, channels=None):
    '''
    read waveforms from recording
    n_times : waveform temporal length 
    channels : channels to read from 
    '''
    dtype = np.dtype(dtype)
    if n_times is None:
        n_times = spike_size

    # n_times needs to be odd
    if n_times % 2 == 0:
        n_times += 1

    # read all channels
    if channels is None:
        channels = np.arange(geom_array.shape[0])
    n_channels = len(channels)

    # ***** LOAD RAW RECORDING *****
    wfs = np.zeros((len(spike_times), n_times, len(channels)),
                   'float32')

    skipped_idx = []
    total_size = n_times*n_channels
    # spike_times are the centers of waveforms
    spike_times_shifted = spike_times - n_times//2
    offsets = spike_times_shifted.astype('int64')*dtype.itemsize*n_channels
    with open(bin_file, "rb") as fin:
        for ctr, spike in enumerate(spike_times_shifted):
            try:
                fin.seek(offsets[ctr], os.SEEK_SET)
                wf = np.fromfile(fin,
                                 dtype=dtype,
                                 count=total_size)
                wfs[ctr] = wf.reshape(
                    n_times, n_channels)[:,channels]
            except:
                skipped_idx.append(ctr)
    wfs=np.delete(wfs, skipped_idx, axis=0)
    fin.close()

    return wfs, skipped_idx

def get_ptp(wfs, templates, unit, chan):
    argmin_idx = templates[unit, :, chan].argmin()
    argmax_idx = templates[unit, :, chan].argmax()
    return(wfs[:, argmax_idx, chan] - wfs[:, argmin_idx, chan])

def GCS(Y, Z):
    if Z.shape[0] < 3:
        return 0
    if len(Z.shape) == 1:
        Z = Z[:,np.newaxis]
    Y += np.random.randn(Y.shape[0])*1e-6
    Z += np.random.randn(Z.shape[0], Z.shape[1])*1e-6
    n = Y.shape[0]
    R = np.argsort(Y)
    tree = KDTree(Z)
    _, ind = tree.query(Z, k=3)
    Ri = np.zeros(Y.shape[0])
    Ri[R] = np.arange(Y.shape[0])+1
    Li = n - Ri + 1
    return (n * np.minimum(Ri, Ri[ind[:,1]]) - Li**2).sum()/(Li * (n-Li)).sum()

def get_ptp_dis_gcs(spike_train, reader, geom_array, bin_file, dtype, displacement, num_obs = 2000):
    num_unit = spike_train[:, 1].max()+1
    coef_per_unit = np.zeros(num_unit)
    for unit in tqdm(np.unique(spike_train[:, 1])):

        spike_times = spike_train[spike_train[:, 1]==unit, 0]
        idx = np.random.choice(np.arange(0, spike_times.shape[0]), size=min(num_obs, spike_times.shape[0]), replace=False)
        spike_times = spike_times[idx]
        # spike_times = np.unique(spike_times)
        wfs, kept_idx = read_waveforms(spike_times, geom_array, bin_file, dtype)
        mc = templates[unit].ptp(0).argmax()

        mask = np.ones(spike_times.shape[0], bool)
        mask[kept_idx] = False
        spike_times = spike_times[mask]

        ptp_unit = get_ptp(wfs, templates, unit, mc)
        #### get ptp displacement correlation
        idx_displacement = spike_times//30000
        displacement_unit = displacement[idx_displacement]
        coef_per_unit[unit] = GCS(ptp_unit, displacement_unit)
    return coef_per_unit


##### Load data, spike train, and displacement
fname_displacement = 'displacement.npy'
displacement = np.load(fname_displacement)
fname_templates = 'templates.npy'
templates = np.load(fname_templates)
fname_spike_train = 'spike_train.npy'
spike_train = np.load(fname_spike_train)
fname_geom_array = 'geom_array.npy'
geom_array = np.load(fname_geom_array)
bin_file = 'standardized.bin'
dtype = 'float32' #Type of standardized bin file

### The following code is written specifically for IBL datasets

# Set align_times to be the beginning of each trial
align_times = 
# Make sure align_times is in seconds (or same time resolution as displacement)
displacement_align_times = displacement[align_times]

#### Get correlation PTP/Displacement
gcs_ptp_displacement = get_ptp_dis_gcs(spike_train, reader, geom_array, bin_file, dtype, displacement)

#### Get correlation FR/Displacement
fr_units = calculate_peths(spike_train[:, 0], spike_train[:, 1], np.unique( spike_train[:, 1]), align_times, pre_time=0.2, post_time=0, bin_size=0.2, smoothing=0, return_fr=True)[1]

gcs_fr_displacement = np.zeros(fr_units.shape[0])
for unit in range(fr_units.shape[0]):
    displacement_unit = 
    gcs_fr_displacement[unit] = GCS(fr_units[unit], displacement_align_times)


#### PCA of fr_units / GCS with displacement 

pca_model = PCA(n_components=1)
neural_activity_pca = pca_model.fit_transform(pca_model.T)

gcs_neural_activity_displacement = GCS(neural_activity_pca, displacement_align_times)

### Scatter plots gcs_fr_displacement vs. gcs_ptp_displacement
### gcs_ptp_displacement vs ptp 
### gcs_fr_displacement vs mean fr

