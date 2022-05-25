"""
Detection pipeline
"""
import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import parmap

from detect.detector import Detect
from localization_pipeline.denoiser import Denoise

from detect.deduplication import deduplicate_gpu, deduplicate

from scipy.signal import argrelmin
from scipy.spatial.distance import pdist, squareform

#### ADD ARGUMENTS AND CHANGE ARGUMENTS

def voltage_threshold(recording, threshold, order=5):

    T, C = recording.shape
    spike_index = np.zeros((0, 2), 'int32')
    energy = np.zeros(0, 'float32')
    for c in range(C):
        single_chan_rec = recording[:, c]
        index = argrelmin(single_chan_rec, order=order)[0]
        index = index[single_chan_rec[index] < -threshold]
        spike_index_temp = np.vstack((index,
                                      np.ones(len(index), 'int32')*c)).T
        spike_index = np.concatenate((spike_index, spike_index_temp), axis=0)
        energy_ = np.abs(single_chan_rec[index])
        energy = np.hstack((energy, energy_))

    return spike_index, energy

def gather_result(fname_save, batch_files_dir):

    logger = logging.getLogger(__name__)
    logger.info('gather detected spikes')

    fnames = os.listdir(batch_files_dir)
    spike_index = []
    spike_index_prekill = []
    n_spikes_prekill = 0
    n_spikes_postkill = 0
    for batch_id in range(len(fnames)):

        # detection index
        fname = os.path.join(batch_files_dir, fnames[batch_id])
        detect_data =  np.load(fname, allow_pickle=True)
        spike_index_prekill_list = detect_data['spike_index']
        spike_index_list = detect_data['spike_index_dedup']
        minibatch_loc = detect_data['minibatch_loc']

        # kill edge spikes and gather results
        for ctr in range(len(spike_index_list)):

            t_start, t_end = minibatch_loc[ctr]
            spike_index_temp = spike_index_list[ctr]

            idx_keep = np.where(np.logical_and(
                spike_index_temp[:, 0] >= t_start,
                spike_index_temp[:, 0] < t_end))[0]
            spike_index.append(spike_index_temp[idx_keep].astype('int32'))

            spike_index_prekill.append(spike_index_prekill_list[ctr].astype('int32'))

            n_spikes_prekill += spike_index_prekill_list[ctr].shape[0]
            n_spikes_postkill += len(idx_keep)

    logger.info('Total {} spikes detected'.format(
        n_spikes_prekill))
    logger.info('Total {} spikes survived after deduplication'.format(
        n_spikes_postkill))

    spike_index = np.vstack(spike_index)
    #spike_index_prekill = np.vstack(spike_index_prekill)
    
    idx_sort = np.argsort(spike_index[:,0])
    spike_index = spike_index[idx_sort]

    #idx_sort = np.argsort(spike_index_prekill[:,0])
    #spike_index_prekill = spike_index_prekill[idx_sort]

    np.save(fname_save, spike_index)
    #np.save(fname_save[:fname_save.rfind('.')]+'_prekill.npy',
    #        spike_index_prekill)

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
            right_buffer_size = int(data_end - rec_len*sampling_rate)
            data_end = int(rec_len*sampling_rate)
        else:
            right_buffer_size = int(0)

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

    return data

def read_data_batch_batch(bin_file, batch_id, n_sec_chunk_small, sampling_rate, buffer_size, dtype_str, geom_array, len_recording, spike_size_nn, add_buffer=False, channels=None):
    '''
    this is for nn detection using gpu
    get a batch and then make smaller batches
    '''
    dtype = np.dtype(dtype_str)
    data = read_data_batch(bin_file, batch_id, len_recording, sampling_rate, n_sec_chunk_small, spike_size_nn, geom_array, dtype, add_buffer, channels) ## ADD ARGUMENTS
    
    T, C = data.shape
    T_mini = int(sampling_rate*n_sec_chunk_small)

    if add_buffer:
        T = T - 2*buffer_size
    else:
        buffer_size = 0

    # # batch sizes
    # indexes = np.arange(0, T, T_mini)
    # indexes = np.hstack((indexes, T))
    # indexes += buffer


    indexes = np.arange(0, T, T_mini)
    indexes = np.hstack((indexes, indexes[-1]+T_mini))
    indexes += buffer_size
   
   
    n_mini_batches = len(indexes) - 1
    # add addtional buffer if needed
    if n_mini_batches*T_mini > T:
        T_extra = n_mini_batches*T_mini - T

        pad_zeros = np.zeros((T_extra, C),
            dtype=dtype)

        data = np.concatenate((data, pad_zeros), axis=0)
    data_loc = np.zeros((n_mini_batches, 2), 'int32')
    data_batched = np.zeros((n_mini_batches, T_mini + 2*buffer_size, C), 'float32')
    for k in range(n_mini_batches):
        data_batched[k] = data[indexes[k]-buffer_size:indexes[k+1]+buffer_size]
        data_loc[k] = [indexes[k], indexes[k+1]]
    return data_batched, data_loc

def make_channel_index(neighbors, channel_geometry, steps=1):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C, C2 = neighbors.shape

    if C != C2:
        raise ValueError('neighbors is not a square matrix, verify')

    # get neighbors matrix
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # FIXME: we are using C as a dummy value which is confusing, it may
    # be better to use something else, maybe np.nan
    # initialize channel index, initially with a dummy C value (a channel)
    # that does not exists
    channel_index = np.ones((C, n_neighbors), 'int32') * C

    # fill every row in the matrix (one per channel)
    for current in range(C):

        # indexes of current channel neighbors
        neighbor_channels = np.where(neighbors[current])[0]

        # sort them by distance
        ch_idx, _ = order_channels_by_distance(current, neighbor_channels,
                                               channel_geometry)

        # fill entries with the sorted neighbor indexes
        channel_index[current, :ch_idx.shape[0]] = ch_idx

    return channel_index

def find_channel_neighbors(geom, radius):
    """Compute a neighbors matrix by using a radius
    Parameters
    ----------
    geom: np.array
        Array with the cartesian coordinates for the channels
    radius: float
        Maximum radius for the channels to be considered neighbors
    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    return (squareform(pdist(geom)) <= radius)
def make_channel_groups(n_channels, neighbors, geom):
    """[DESCRIPTION]
    Parameters
    ----------
    n_channels: int
        Number of channels
    neighbors: numpy.ndarray
        Neighbors matrix
    geom: numpy.ndarray
        geometry matrix
    Returns
    -------
    list
        List of channel groups based on [?]
    """
    channel_groups = list()
    c_left = np.array(range(n_channels))
    neighChan_temp = np.array(neighbors)

    while len(c_left) > 0:
        c_tops = c_left[geom[c_left, 1] == np.max(geom[c_left, 1])]
        c_topleft = c_tops[np.argmin(geom[c_tops, 0])]
        c_group = np.where(
            np.sum(neighChan_temp[neighChan_temp[c_topleft]], 0))[0]

        neighChan_temp[c_group, :] = 0
        neighChan_temp[:, c_group] = 0

        for c in c_group:
            c_left = np.delete(c_left, int(np.where(c_left == c)[0]))

        channel_groups.append(c_group)

    return channel_groups


def order_channels_by_distance(reference, channels, geom):
    """Order channels by distance using certain channel as reference
    Parameters
    ----------
    reference: int
        Reference channel
    channels: np.ndarray
        Channels to order
    geom
        Geometry matrix
    Returns
    -------
    numpy.ndarray
        1D array with the channels ordered by distance using the reference
        channels
    numpy.ndarray
        1D array with the indexes for the ordered channels
    """
    coord_main = geom[reference]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))

    return channels[idx], idx


def n_steps_neigh_channels(neighbors_matrix, steps):
    """Compute a neighbors matrix by considering neighbors of neighbors
    Parameters
    ----------
    neighbors_matrix: numpy.ndarray
        Neighbors matrix
    steps: int
        Number of steps to still consider channels as neighbors
    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    C = neighbors_matrix.shape[0]

    # each channel is its own neighbor (diagonal of trues)
    output = np.eye(C, dtype='bool')

    # for every step
    for _ in range(steps):

        # go trough every channel
        for current in range(C):

            # neighbors of the current channel
            neighbors_current = output[current]

            # get the neighbors of all the neighbors of the current channel
            neighbors_of_neighbors = neighbors_matrix[neighbors_current]

            # sub over rows and convert to bool, this will turn to true entries
            # where at least one of the neighbors has each channel as its
            # neighbor
            is_neighbor_of_neighbor = np.sum(neighbors_of_neighbors,
                                             axis=0).astype('bool')

            # set the channels that are neighbors to true
            output[current][is_neighbor_of_neighbor] = True

    return output

def run(standardized_path, standardized_dtype, output_directory, geom_array, spatial_radius, apply_nn, n_sec_chunk, n_batches, n_processors, n_sec_chunk_gpu_detect, sampling_rate, len_recording,
    detect_threshold, path_nn_detector, n_filters_detect, spike_size_nn, path_nn_denoiser, n_filters_denoise, filter_sizes_denoise, run_chunk_sec='full'):
           
    """Execute detect step
    Parameters
    ----------
    standardized_path: str or pathlib.Path
        Path to standardized data binary file
    standardized_dtype: string
        data type of standardized data
    output_directory: str, optional
      Location to store partial results, relative to CONFIG.data.root_folder,
      defaults to tmp/
    Returns
    -------
    clear_scores: numpy.ndarray (n_spikes, n_features, n_channels)
        3D array with the scores for the clear spikes, first simension is
        the number of spikes, second is the nymber of features and third the
        number of channels
    spike_index_clear: numpy.ndarray (n_clear_spikes, 2)
        2D array with indexes for clear spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    spike_index_call: numpy.ndarray (n_collided_spikes, 2)
        2D array with indexes for all spikes, first column contains the
        spike location in the recording and the second the main channel
        (channel whose amplitude is maximum)
    Notes
    -----
    Running the preprocessor will generate the followiing files in
    CONFIG.data.root_folder/output_directory/ (if save_results is
    True):
    * ``spike_index_clear.npy`` - Same as spike_index_clear returned
    * ``spike_index_all.npy`` - Same as spike_index_collision returned
    * ``rotation.npy`` - Rotation matrix for dimensionality reduction
    * ``scores_clear.npy`` - Scores for clear spikes
    Threshold detector runs on CPU, neural network detector runs CPU and GPU,
    depending on how pytorch is configured.
    Examples
    --------
    .. literalinclude:: ../../examples/pipeline/detect.py
    """
    
    # make output directory if not exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    fname_spike_index = os.path.join(
        output_directory, 'spike_index.npy')
    if os.path.exists(fname_spike_index):
        print("Detection already done!")
        return fname_spike_index

    ##### detection #####
    # save directory for temp files
    output_temp_files = os.path.join(
        output_directory, 'batch')
    if not os.path.exists(output_temp_files):
        os.mkdir(output_temp_files)

    # neighboring channels
    neigh_channels = find_channel_neighbors(geom_array, spatial_radius)
    channel_index = make_channel_index(neigh_channels, geom_array)
    # run detection
    if apply_nn:
        ## CHANGE ARGUMENTS
        run_neural_network(
            standardized_path,
            standardized_dtype,
            output_temp_files,
            sampling_rate, 
            len_recording, 
            n_sec_chunk,
            n_processors, 
            n_sec_chunk_gpu_detect, 
            detect_threshold,
            neigh_channels, 
            geom_array,
            path_nn_detector, 
            n_filters_detect, 
            spike_size_nn, 
            path_nn_denoiser, 
            n_filters_denoise, 
            filter_sizes_denoise,
            run_chunk_sec=run_chunk_sec)

    else:
        run_voltage_treshold(standardized_path,
                             standardized_dtype,
                             spike_size_nn,
                             n_sec_chunk, 
                             geom_array,
                             neigh_channels, 
                             n_batches, ### N_BATCHES
                             detect_threshold, 
                             channel_index,
                             output_temp_files,
                             sampling_rate,
                             len_recording, 
                             run_chunk_sec=run_chunk_sec)

##### gather results #####
    gather_result(fname_spike_index,
                  output_temp_files)

    return fname_spike_index


def run_neural_network(standardized_path, standardized_dtype, output_directory, sampling_rate, len_recording, n_sec_chunk, n_processors, n_sec_chunk_gpu_detect, detect_threshold,
    neigh_channels, geom_array, path_nn_detector, n_filters_detect, spike_size_nn, path_nn_denoiser, n_filters_denoise, filter_sizes_denoise, run_chunk_sec='full'):
                           
    """Run neural network detection
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # load NN detector
    channel_index = make_channel_index(neigh_channels, geom_array, steps=1)
    detector = Detect(n_filters_detect, spike_size_nn, channel_index)
    detector.load(path_nn_detector)

    # load NN denoiser
    denoiser = Denoise(n_filters_denoise,
                       filter_sizes_denoise,
                       spike_size_nn)
    denoiser.load(path_nn_denoiser)

    # get data reader
    n_batches = len_recording//n_sec_chunk
    batch_length = n_sec_chunk*n_processors
    n_sec_chunk = n_sec_chunk_gpu_detect
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer_size = spike_size_nn
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec
    
    # neighboring channels

    channel_index_dedup = make_channel_index(
        neigh_channels, geom_array, steps=2)

    # loop over each chunk
    batch_ids = np.arange(n_batches)
    
    if False:
        batch_ids_split = np.split_array(batch_ids, len(CONFIG.torch_devices))
        processes = []
        for ii, device in enumerate([torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]): ### SEVERAL DEVICES??
            p = mp.Process(target=run_nn_detection_batch,
                           args=(standardized_path, batch_ids_split[ii], output_directory, n_sec_chunk, spike_size_nn, geom_array, 
                                 sampling_rate, len_recording, buffer_size, standardized_dtype,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        run_nn_detection_batch(standardized_path, batch_ids, output_directory, n_sec_chunk, spike_size_nn, geom_array, sampling_rate, len_recording, buffer_size, standardized_dtype,
                                 detector, denoiser, channel_index_dedup,
                                 detect_threshold, device=0) #CONFIG.resources.gpu_id?


def run_nn_detection_batch(standardized_path, batch_ids, output_directory, n_sec_chunk, spike_size_nn, geom_array, 
                          sampling_rate, len_recording, buffer_size, dtype_str,
                          detector, denoiser,
                          channel_index_dedup,
                          detect_threshold,
                          device):

    detector = detector.to(device)
    denoiser = denoiser.to(device)
    ### GET IDX LIST
    idx_list, n_batches = get_idx_list_n_batches(n_sec_chunk, sampling_rate, 0, len_recording)

    for batch_id in batch_ids:
        # skip if the file exists
        fname = os.path.join(
            output_directory,
            "detect_" + str(batch_id).zfill(5) + '.npz')

        if os.path.exists(fname):
            continue


        # get a bach of size n_sec_chunk
        # but partioned into smaller minibatches of 
        # size n_sec_chunk_gpu
        ### UPDATE ARGUMENTS
        batched_recordings, minibatch_loc_rel = read_data_batch_batch(standardized_path, batch_id, n_sec_chunk, sampling_rate, buffer_size, dtype_str, geom_array, len_recording, spike_size_nn, add_buffer=True)
        # offset for big batch
        batch_offset = idx_list[batch_id, 0] - buffer_size
        # location of each minibatch (excluding buffer)
        minibatch_loc = minibatch_loc_rel + batch_offset
        spike_index_list = []
        spike_index_dedup_list = []
        for j in range(batched_recordings.shape[0]):
            # detect spikes and get wfs
            spike_index, wfs = detector.get_spike_times(
                torch.FloatTensor(batched_recordings[j]).to(device),
                threshold=detect_threshold)

            # denoise and take ptp as energy
            if len(spike_index) == 0:
                del spike_index, wfs
                continue

            wfs_denoised = denoiser(wfs)[0].data
            energy = (torch.max(wfs_denoised, 1)[0] - torch.min(wfs_denoised, 1)[0])

            # deduplicate
            spike_index_dedup = deduplicate_gpu(
                spike_index, energy,
                batched_recordings[j].shape,
                channel_index_dedup)

            # convert to numpy
            spike_index_cpu = spike_index.cpu().data.numpy()
            spike_index_dedup_cpu = spike_index_dedup.cpu().data.numpy()

            # update the location relative to the whole recording
            spike_index_cpu[:, 0] += (minibatch_loc[j, 0] - buffer_size)
            spike_index_dedup_cpu[:, 0] += (minibatch_loc[j, 0] - buffer_size)
            spike_index_list.append(spike_index_cpu)
            spike_index_dedup_list.append(spike_index_dedup_cpu)

            del wfs
            del wfs_denoised
            del energy
            del spike_index
            del spike_index_dedup

            torch.cuda.empty_cache()

        #if processing_ctr%100==0:
        print('batch : {}'.format(batch_id))

        # save result
        np.savez(fname,
                 spike_index=spike_index_list,
                 spike_index_dedup=spike_index_dedup_list,
                 minibatch_loc=minibatch_loc)
        
    del detector
    del denoiser


def run_voltage_treshold(standardized_path, standardized_dtype, spike_size_nn, n_sec_chunk, geom_array, neigh_channels, n_batches, threshold, channel_index,
                         output_directory, sampling_rate, len_recording, run_chunk_sec='full'):
                           
    """Run detection that thresholds on amplitude
    """
    logger = logging.getLogger(__name__)

    # get data reader
    #n_sec_chunk = CONFIG.resources.n_sec_chunk*CONFIG.resources.n_processors
    ### ADD Arguments
    batch_length = n_sec_chunk
    n_sec_chunk = 0.5
    print ("   batch length to (sec): ", batch_length, 
           " (longer increase speed a bit)")
    print ("   length of each seg (sec): ", n_sec_chunk)
    buffer_size = spike_size_nn
    if run_chunk_sec == 'full':
        chunk_sec = None
    else:
        chunk_sec = run_chunk_sec

    # reader = READER(standardized_path,
    #                 standardized_dtype,
    #                 CONFIG,
    #                 batch_length,
    #                 buffer,
    #                 chunk_sec)

    # number of processed chunks
    n_mini_per_big_batch = int(np.ceil(batch_length/n_sec_chunk))    
    total_processing = int(n_batches*n_mini_per_big_batch)

    # neighboring channels
    channel_index = make_channel_index(
        neigh_channels, geom_array, steps=2)
    idx_list, n_batches = get_idx_list_n_batches(n_sec_chunk, sampling_rate, 0, len_recording)

    ## ADD ARGUMENTS
    multi_processing_flag = True
    n_processors = 4
    if multi_processing_flag:
        parmap.starmap(run_voltage_threshold_parallel, 
                       list(zip(np.arange(n_batches))),
                       n_sec_chunk,
                       threshold,
                       channel_index,
                       buffer_size, 
                       standardized_dtype, 
                       idx_list,
                       output_directory,
                       sampling_rate, standardized_path, geom_array, len_recording, spike_size_nn, 
                       processes=n_processors,
                       pm_pbar=True)                
    else:
        for batch_id in range(n_batches):
            run_voltage_threshold_parallel(
                batch_id,
                n_sec_chunk,
                threshold,
                channel_index,
                buffer_size,
                standardized_dtype, 
                idx_list,
                output_directory, sampling_rate, standardized_path, geom_array, len_recording, spike_size_nn)


def run_voltage_threshold_parallel(batch_id, n_sec_chunk,
                                   threshold, channel_index, buffer_size, dtype_str, idx_list,
                                   output_directory, sampling_rate, bin_file, geom_array, len_recording, spike_size_nn):

    # skip if the file exists
    fname = os.path.join(
        output_directory,
        "detect_" + str(batch_id).zfill(5) + '.npz')

    if os.path.exists(fname):
        return

    # get a bach of size n_sec_chunk
    # but partioned into smaller minibatches of 
    # size n_sec_chunk_gpu        

    batched_recordings, minibatch_loc_rel = read_data_batch_batch(
        bin_file, 
        batch_id,
        n_sec_chunk,
        sampling_rate, 
        buffer_size, 
        dtype_str, 
        geom_array, len_recording, spike_size_nn,
        add_buffer=True)

    # offset for big batch
    batch_offset = idx_list[batch_id, 0] - buffer_size
    # location of each minibatch (excluding buffer)
    minibatch_loc = minibatch_loc_rel + batch_offset
    spike_index_list = []
    spike_index_dedup_list = []
    for j in range(batched_recordings.shape[0]):
        spike_index, energy = voltage_threshold(
            batched_recordings[j], 
            threshold)

        # move to gpu
        spike_index = torch.from_numpy(spike_index)
        energy = torch.from_numpy(energy)

        # deduplicate
        spike_index_dedup = deduplicate_gpu(
            spike_index, energy,
            batched_recordings[j].shape,
            channel_index)

        # convert to numpy
        spike_index = spike_index.cpu().data.numpy()
        spike_index_dedup = spike_index_dedup.cpu().data.numpy()
        
        # update the location relative to the whole recording
        spike_index[:, 0] += (minibatch_loc[j, 0] - buffer_size)
        spike_index_dedup[:, 0] += (minibatch_loc[j, 0] - buffer_size)
        spike_index_list.append(spike_index)
        spike_index_dedup_list.append(spike_index_dedup)

    # save result
    np.savez(fname,
             spike_index=spike_index_list,
             spike_index_dedup=spike_index_dedup_list,
             minibatch_loc=minibatch_loc)