import os
import numpy as np
import parmap
import yaml
import numpy as np
from scipy.signal import butter, filtfilt

def get_idx_list_n_batches(n_sec_chunk, sampling_rate, start, end):
    batch_size = sampling_rate*n_sec_chunk
    indexes = np.arange(start*sampling_rate, end*sampling_rate, batch_size)
    indexes = np.hstack((indexes, end*sampling_rate))

    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([indexes[k], indexes[k + 1]])
    idx_list = np.int64(np.vstack(idx_list))
    n_batches = len(idx_list)
    return idx_list, n_batches

def read_data(bin_file, dtype_str, data_start, data_end, geom_array, channels=None):
    dtype = np.dtype(dtype_str)
    if channels is None:
        n_channels = geom_array.shape[0]
    else:
        n_channels = len(channels)
    with open(bin_file, "rb") as fin:

        fin.seek(int((data_start)*dtype.itemsize*n_channels), os.SEEK_SET)
        data = np.fromfile(
            fin, dtype=dtype,
            count=int((data_end - data_start)*n_channels))
    fin.close()
    
    data = data.reshape(-1, n_channels)
    if channels is not None:
        data = data[:, channels]

    return data

def read_data_batch(bin_file, batch_id, rec_len, sampling_rate, n_sec_chunk, spike_size_nn, geom_array, dtype_str, add_buffer=False, channels=None):
    dtype = np.dtype(dtype_str)
    batch_size = sampling_rate*n_sec_chunk

    idx_list, n_batches = get_idx_list_n_batches(n_sec_chunk, sampling_rate, 0, rec_len)

    # batch start and end
    data_start, data_end = idx_list[batch_id]
    # add buffer if asked
    buffer_templates = spike_size_nn
    if add_buffer:
        data_start -= buffer_templates
        data_end += buffer_templates

        # if start is below zero, put it back to 0 and and zeros buffer
        if data_start < 0:
            left_buffer_size = - data_start
            data_start = 0
        else:
            left_buffer_size = 0

        # if end is above rec_len, put it back to rec_len and and zeros buffer
        if data_end > rec_len*sampling_rate:
            right_buffer_size = data_end - rec_len*sampling_rate
            data_end = rec_len*sampling_rate
        else:
            right_buffer_size = 0

    #data_start= int(data_start)
    #data_end = int(data_end)
    # read data
    data = read_data(bin_file, dtype_str, data_start, data_end, geom_array, channels)
    # add leftover buffer with zeros if necessary
    if channels is None:
        n_channels = geom_array.shape[0]
    else:
        n_channels = len(channels)

    if add_buffer:
        left_buffer = np.zeros(
            (left_buffer_size, n_channels),
            dtype=dtype)
        right_buffer = np.zeros(
            (right_buffer_size, n_channels),
            dtype=dtype)
        if channels is not None:
            left_buffer = left_buffer[:, channels]
            right_buffer = right_buffer[:, channels]
        data = np.concatenate((left_buffer, data, right_buffer), axis=0)
    else:
        left_buffer_size = 0 
        right_buffer_size = 0

    return data, buffer_templates, buffer_templates

def _butterworth(ts, low_frequency, high_factor, order, sampling_frequency):
    """Butterworth filter
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    low = float(low_frequency) / sampling_frequency * 2
    high = float(high_factor) * 2
    b, a = butter(order, low, btype='high', analog=False)

    if ts.ndim == 1:
        return filtfilt(b, a, ts)
    else:
        T, C = ts.shape
        output = np.zeros((T, C), 'float32')
        for c in range(C):
            output[:, c] = filtfilt(b, a, ts[:, c])

        return output


def _mean_standard_deviation(rec, centered=False):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
    centered : bool
        if not standardized, center it
    Returns
    -------
    sd : vector [number of channels]
        standard deviation in each channel
    """

    # find standard deviation using robust method
    if not centered:
        centers = np.mean(rec, axis=0)
        rec = rec - centers[None]
    else:
        centers = np.zeros(rec.shape[1], 'float32')

    return np.median(np.abs(rec), 0)/0.6745, centers


def _standardize(rec, sd=None, centers=None):
    """Determine standard deviation of noise in each channel
    Parameters
    ----------
    rec : matrix [length of recording, number of channels]
        recording
    sd : vector [number of chnanels,]
        standard deviation
    centered : bool
        if not standardized, center it
    Returns
    -------
    matrix [length of recording, number of channels]
        standardized recording
    """

    # find standard deviation using robust method
    if (sd is None) or (centers is None):
        sd, centers = _mean_standard_deviation(rec, centered=False)

    # standardize all channels with SD> 0.1 (Voltage?) units
    # Cat: TODO: ensure that this is actually correct for all types of channels
    idx1 = np.where(sd>=0.1)[0]
    rec[:,idx1] = np.divide(rec[:,idx1] - centers[idx1][None], sd[idx1])
    
    # zero out bad channels
    idx2 = np.where(sd<0.1)[0]
    rec[:,idx2]=0.
    
    return rec
    #return np.divide(rec, sd)


def filter_standardize_batch(batch_id, bin_file, dtype_str, rec_len, sampling_rate, fname_mean_sd, n_sec_chunk, spike_size_nn, geom_array, 
                             apply_filter, out_dtype, output_directory, 
                             low_frequency=None, high_factor=None,
                             order=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    
    # filter
    if apply_filter:
        # read a batch
        ts, left_buffer_size, right_buffer_size = read_data_batch(bin_file, batch_id, rec_len, sampling_rate, n_sec_chunk, spike_size_nn, geom_array, dtype_str, add_buffer=True)
        ts = _butterworth(ts, low_frequency, high_factor,
                              order, sampling_rate)
        ts = ts[left_buffer_size:ts.shape[0]-right_buffer_size]
    else:
        ts, left_buffer_size, right_buffer_size = read_data_batch(bin_file, batch_id, rec_len, sampling_rate, n_sec_chunk, spike_size_nn, geom_array, dtype_str, add_buffer=False)

    # standardize
    temp = np.load(fname_mean_sd)
    sd = temp['sd']
    centers = temp['centers']
    ts = _standardize(ts, sd, centers)
    
    # save
    fname = os.path.join(
        output_directory,
        "standardized_{}.npy".format(
            str(batch_id).zfill(6)))
    np.save(fname, ts.astype(out_dtype))

    #fname = os.path.join(
    #    output_directory,
    #    "standardized_{}.bin".format(
    #        str(batch_id).zfill(6)))
    #f = open(fname, 'wb')
    #f.write(ts.astype(out_dtype))


def get_std(ts,
            sampling_frequency,
            fname,
            apply_filter=False, 
            low_frequency=None,
            high_factor=None,
            order=None):
    """Butterworth filter for a one dimensional time series
    Parameters
    ----------
    ts: np.array
        T  numpy array, where T is the number of time samples
    low_frequency: int
        Low pass frequency (Hz)
    high_factor: float
        High pass factor (proportion of sampling rate)
    order: int
        Order of Butterworth filter
    sampling_frequency: int
        Sampling frequency (Hz)
    Notes
    -----
    This function can only be applied to a one dimensional array, to apply
    it to multiple channels use butterworth
    Raises
    ------
    NotImplementedError
        If a multidmensional array is passed
    """

    # filter
    if apply_filter:
        ts = _butterworth(ts, low_frequency, high_factor,
                          order, sampling_frequency)

    # standardize
    sd, centers = _mean_standard_deviation(ts)
    
    # save
    np.savez(fname,
             centers=centers,
             sd=sd)


def merge_filtered_files(filtered_location, output_directory):

    filenames = os.listdir(filtered_location)
    filenames_sorted = sorted(filenames)

    f_out = os.path.join(output_directory, "standardized.bin")

    f = open(f_out, 'wb')
    for fname in filenames_sorted:
        res = np.load(os.path.join(filtered_location, fname))
        res.tofile(f)
        os.remove(os.path.join(filtered_location, fname))

#### PARAMS ####
raw_data_path = 'non_standardized.bin'
dtype_raw = 'int16'
sampling_rate = 30000
rec_len = 1000 #length of recording in seconds
geom_array = np.load('geom.npy') #geometry of channels

output_directory = 'standardized_data_dir'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


n_sec_chunk = 1
preprocess_dtype = 'float32' #Output will be of this type

##recommended for most NP probes
apply_filter = True
low_frequency = 300
high_factor = 0.1
order = 3

spike_size_nn = 121 # Length of spikes - here, 4 ms with 30000 sampling rate -> 121 timesteps

# estimate std from a small chunk
chunk_5sec = 5*sampling_rate
if rec_len < chunk_5sec:
    chunk_5sec = rec_len
data_start=rec_len//2 - chunk_5sec//2
data_end=rec_len//2 + chunk_5sec//2
small_batch = read_data(raw_data_path, dtype_raw, data_start, data_end, geom_array)


fname_mean_sd = os.path.join(
    output_directory, 'mean_and_standard_dev_value.npz')


if not os.path.exists(fname_mean_sd):
    get_std(small_batch, sampling_rate,
            fname_mean_sd, apply_filter,
            low_frequency, high_factor, order)


# Make directory to hold filtered batch files:
filtered_location = os.path.join(output_directory, "filtered_files")
if not os.path.exists(filtered_location):
    os.makedirs(filtered_location)

# read config params
multi_processing = True #set to True for multi processing, False otherwise
n_processors = 4 #number of processors for multiproce

idx_list, n_batches = get_idx_list_n_batches(n_sec_chunk, sampling_rate, 0, rec_len)

if multi_processing:
    parmap.map(
        filter_standardize_batch,
        [i for i in range(n_batches)],
        raw_data_path,
        dtype_raw, 
        rec_len, 
        sampling_rate,
        fname_mean_sd,
        n_sec_chunk, 
        spike_size_nn, 
        geom_array,
        apply_filter, 
        preprocess_dtype,
        filtered_location,
        low_frequency,
        high_factor,
        order,
        processes=n_processors,
        pm_pbar=True)
else:
    for batch_id in range(n_batches):
        filter_standardize_batch(
            batch_id, raw_data_path,
            dtype_raw, 
            rec_len, 
            sampling_rate,
            fname_mean_sd,
            n_sec_chunk, 
            spike_size_nn, 
            geom_array,
            apply_filter, 
            preprocess_dtype,
            filtered_location,
            low_frequency,
            high_factor,
            order)

# Merge the chunk filtered files and delete the individual chunks
merge_filtered_files(filtered_location, output_directory)

