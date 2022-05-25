import os
import numpy as np
from tqdm import tqdm
# from residual import RESIDUAL
# from localizer import LOCALIZER
# from denoiser import Denoise

def get_total_len(input_directory, n_batches):
    len_total = 0
    len_batch = np.zeros(n_batches)
    for batch_id in tqdm(range(n_batches)):
        fname_z = os.path.join(input_directory, 'results_z_mean_{}.npy'.format(str(batch_id).zfill(6)))
        len_batch[batch_id] = np.load(fname_z).shape[0]
        len_total += len_batch[batch_id]
    return int(len_total), len_batch.astype('int')

def get_merged_arrays(input_directory, output_directory, n_batches):
    len_total, len_batch = get_total_len(input_directory, n_batches)
    
    merged_z_array = np.zeros(len_total)
    merged_z_mean_array = np.zeros(len_total)
    merged_x_array = np.zeros(len_total)
    merged_x_mean_array = np.zeros(len_total)
    merged_max_ptp_array = np.zeros(len_total)
    merged_alpha_array = np.zeros(len_total)
    merged_spread_array = np.zeros(len_total)
    merged_y_array = np.zeros(len_total)
    merged_max_channels = np.zeros(len_total)
    merged_times_read = np.zeros(len_total)
    merged_time_width = np.zeros(len_total)

    cmp = 0
    for batch_id in tqdm(range(n_batches)):
        
        fname_time_width = os.path.join(input_directory, 'results_width_{}.npy'.format(str(batch_id).zfill(6)))
        fname_z = os.path.join(input_directory, 'results_z_{}.npy'.format(str(batch_id).zfill(6)))     
        fname_x = os.path.join(input_directory, 'results_x_{}.npy'.format(str(batch_id).zfill(6)))
        fname_z_mean = os.path.join(input_directory, 'results_z_mean_{}.npy'.format(str(batch_id).zfill(6)))     
        fname_x_mean = os.path.join(input_directory, 'results_x_mean_{}.npy'.format(str(batch_id).zfill(6)))
        fname_spread = os.path.join(input_directory, 'results_spread_{}.npy'.format(str(batch_id).zfill(6)))
        fname_max_ptp = os.path.join(input_directory, 'results_max_ptp_{}.npy'.format(str(batch_id).zfill(6)))
        fname_y = os.path.join(input_directory, 'results_y_{}.npy'.format(str(batch_id).zfill(6)))
        fname_alpha = os.path.join(input_directory, 'results_alpha_{}.npy'.format(str(batch_id).zfill(6)))
        fname_max_channels = os.path.join(input_directory, 'results_max_channels_{}.npy'.format(str(batch_id).zfill(6)))
        fname_times_read = os.path.join(input_directory, 'times_read_{}.npy'.format(str(batch_id).zfill(6)))

        merged_z_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_z)
        merged_z_mean_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_z_mean)
        merged_x_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_x)
        merged_x_mean_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_x_mean)
        merged_max_ptp_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_max_ptp)
        merged_spread_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_spread)
        merged_alpha_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_alpha)
        merged_y_array[cmp:cmp+len_batch[batch_id]] = np.load(fname_y)
        merged_time_width[cmp:cmp+len_batch[batch_id]] = np.load(fname_time_width)
        merged_times_read[cmp:cmp+len_batch[batch_id]] = np.load(fname_times_read)
        merged_max_channels[cmp:cmp+len_batch[batch_id]] = np.load(fname_max_channels)

        cmp+=len_batch[batch_id] 
        
    print(cmp)  

    fname_z_merged = os.path.join(output_directory, 'results_z_merged.npy')
    fname_z_mean_merged = os.path.join(output_directory, 'results_z_mean_merged.npy')
    fname_x_merged = os.path.join(output_directory, 'results_x_merged.npy')
    fname_x_mean_merged = os.path.join(output_directory, 'results_x_mean_merged.npy')
    fname_alpha_merged = os.path.join(output_directory, 'results_alpha_merged.npy')
    fname_y_merged = os.path.join(output_directory, 'results_y_merged.npy')
    fname_max_ptp_merged = os.path.join(output_directory, 'results_max_ptp_merged.npy')
    fname_spread_merged = os.path.join(output_directory, 'results_spread_merged.npy')
    fname_max_channels = os.path.join(output_directory, 'results_max_channels.npy')
    fname_times_read = os.path.join(output_directory, 'times_read.npy')
    fname_time_width = os.path.join(output_directory, 'results_width.npy')

    np.save(fname_z_merged, merged_z_array)
    np.save(fname_x_merged, merged_x_array)
    np.save(fname_alpha_merged, merged_alpha_array)
    np.save(fname_y_merged, merged_y_array)
    np.save(fname_max_ptp_merged, merged_max_ptp_array)
    np.save(fname_spread_merged, merged_spread_array)
    np.save(fname_z_mean_merged, merged_z_mean_array)
    np.save(fname_x_mean_merged, merged_x_mean_array)
    np.save(fname_max_channels, merged_max_channels)
    np.save(fname_times_read, merged_times_read)
    np.save(fname_time_width, merged_time_width)


