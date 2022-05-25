import os
import logging
import numpy as np
import parmap

class RESIDUAL(object):
    
    def __init__(self, 
                 bin_file,
                 fname_templates,
                 fname_spike_train,
                 n_batches,
                 len_recording,
                 sampling_rate,
                 n_channels,
                 fname_out,
                 dtype_in,
                 dtype_out):
        
        """ Initialize by computing residuals
            provide: raw data block, templates, and deconv spike train; 
        """
        #self.logger = logging.getLogger(__name__)

        # keep templates and spike train filname
        # will be loaded during each prallel process
        self.fname_templates = fname_templates
        self.fname_spike_train = fname_spike_train
        self.bin_file = bin_file

        self.n_batches = n_batches

        # save output name and dtype
        self.fname_out = fname_out
        self.dtype_out = dtype_out
        self.sampling_rate = sampling_rate
        self.n_sec_chunk = len_recording // n_batches
        self.dtype = dtype_in
        self.n_channels = n_channels
        
        self.batch_size = self.sampling_rate*self.n_sec_chunk
        indexes = np.arange(0, len_recording*self.sampling_rate, self.batch_size)
        indexes = np.hstack((indexes, len_recording*self.sampling_rate))

        idx_list = []
        for k in range(len(indexes) - 1):
            idx_list.append([indexes[k], indexes[k + 1]])
        self.idx_list = np.int64(np.vstack(idx_list))
        self.n_batches = len(self.idx_list)
        
        filesize = os.path.getsize(self.bin_file)
        dtype = np.dtype(self.dtype)
        self.dtype = dtype
        self.rec_len = int(filesize / 
                           dtype.itemsize / 
                           self.n_channels)

        
    def read_data(self, data_start, data_end, channels=None):
        with open(self.bin_file, "rb") as fin:
            # Seek position and read N bytes
            #fin.seek((data_start-self.offset)*self.dtype.itemsize*self.n_channels, os.SEEK_SET)
            fin.seek(int((data_start)*self.dtype.itemsize*self.n_channels), os.SEEK_SET)
            data = np.fromfile(
                fin, dtype=self.dtype,
                count=int((data_end - data_start)*self.n_channels))
        fin.close()
        
        data = data.reshape(-1, self.n_channels)
        if channels is not None:
            data = data[:, channels]

        return data

    def read_data_batch(self, batch_id, add_buffer=False, channels=None):

        # batch start and end
        data_start, data_end = self.idx_list[batch_id]
        # add buffer if asked
        if add_buffer:
            data_start -= self.buffer
            data_end += self.buffer

            # if start is below zero, put it back to 0 and and zeros buffer
            if data_start < 0:
                left_buffer_size = - data_start
                data_start = 0
            else:
                left_buffer_size = 0

            # if end is above rec_len, put it back to rec_len and and zeros buffer
            len_recording*self.sampling_rate
            if data_end > self.rec_len:
                right_buffer_size = data_end - self.rec_len
                data_end = self.rec_len
            else:
                right_buffer_size = 0

        # read data
        data = self.read_data(data_start, data_end, channels)
        # add leftover buffer with zeros if necessary
        if add_buffer:
            left_buffer = np.zeros(
                (left_buffer_size, self.n_channels),
                dtype=self.dtype)
            right_buffer = np.zeros(
                (right_buffer_size, self.n_channels),
                dtype=self.dtype)
            if channels is not None:
                left_buffer = left_buffer[:, channels]
                right_buffer = right_buffer[:, channels]
            data = np.concatenate((left_buffer, data, right_buffer), axis=0)

        return data



    def compute_residual(self, save_dir,
                         multi_processing=False,
                         n_processors=1):
        '''
        '''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        batch_ids = []
        fnames_seg = []
        for batch_id in range(self.n_batches):
            batch_ids.append(batch_id)
            fnames_seg.append(
                os.path.join(save_dir,
                             'residual_seg{}.npy'.format(batch_id)))

        #self.logger.info("computing residuals")
        if multi_processing:
            batches_in = np.array_split(batch_ids, n_processors)
            fnames_in = np.array_split(fnames_seg, n_processors)
            parmap.starmap(self.subtract_parallel, 
                         list(zip(batches_in, fnames_in)),
                         processes=n_processors,
                         pm_pbar=True)

        else:
            for ctr in range(len(batch_ids)):
                self.subtract_parallel(
                    [batch_ids[ctr]], [fnames_seg[ctr]])

        self.fnames_seg = fnames_seg


    def subtract_parallel(self, batch_ids, fnames_out):
        '''
        '''
                
        templates = None
        spike_train = None

        for batch_id, fname_out in zip(batch_ids, fnames_out):
            if os.path.exists(fname_out):
                continue
            
            # load upsampled templates only once per core:
            if templates is None or spike_train is None:
                #self.logger.info("loading upsampled templates")
                templates = np.load(self.fname_templates)
                spike_train = np.load(self.fname_spike_train)

                # do not read spike train again here
                #self.spike_train = up_data['spike_train_up']
                n_time = templates.shape[1]
                time_idx = np.arange(0, n_time)
                self.buffer = n_time
                
                # shift spike time so that it is aligned at
                # time 0
                spike_train[:, 0] -= n_time//2

            # get relevantspike times
            start, end = self.idx_list[batch_id]
            start -= self.buffer 
            idx_in_chunk = np.where(
                np.logical_and(spike_train[:,0]>=start,
                               spike_train[:,0]<end))[0]
            spikes_in_chunk = spike_train[idx_in_chunk]
            # offset
            spikes_in_chunk[:,0] -= start

            data = self.read_data_batch(batch_id, add_buffer=True)

            for j in range(spikes_in_chunk.shape[0]):
                tt, ii = spikes_in_chunk[j]
                data[time_idx + tt, :] -= templates[ii]            
            
            data = data[self.buffer:-self.buffer]

            # save
            np.save(fname_out, data)


    def save_residual(self):
    
        f = open(self.fname_out,'wb')
        for fname in self.fnames_seg:

            res = np.load(fname).astype(self.dtype_out)
            f.write(res)
        f.close()
        
        # delete residual chunks after successful merging/save
        for fname in self.fnames_seg:
            os.remove(fname)