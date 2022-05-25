import os
import numpy as np
from tqdm import tqdm
from residual import RESIDUAL
from localizer import LOCALIZER
from denoiser import Denoise
from merge_results import get_merged_arrays



### Change paths to your data here
bin_file = 'standardized_data.bin'
dtype_input = 'float32'

fname_spike_train = "spt.npy" 
# Sort spike train if not 
spt_array = np.load(fname_spike_train)
spt_array = spt_array[spt_array[:, 0].argsort()]
np.save(fname_spike_train, spt_array)

geom_path = "channels_maps/np2_channel_map.npy"
n_channels = np.load(geom_path).shape[0]

denoiser_weights = '/pretrained_denoiser/denoise.pt'
denoiser_min = 42 ## Goes with the weights

n_batches = 1000
len_recording = 1000
sampling_rate = 30000

### If templates are already computed, input the path here
fname_templates = "templates.npy"
### Otherwise, uncomment and run 
# fname_templates = None
# localizer_obj = LOCALIZER(bin_file, residual_file, dtype_input, fname_spike_train, fname_templates, geom_path, denoiser_weights, denoiser_min)
# localizer_obj.get_templates()
# fname_templates = "templates.npy"
# np.save(fname_templates, localizer_obj.templates)

##### COMPUTE RESIDUALS 
'''
This is necessary for denoising spikes and removing collisions
It needs to be ran only once per datasets, as residual.bin file can be stored and reused 
'''

fname_out = 'residuals/residual.bin'
dtype_out = 'float32'
dtype_in = 'float32'

residual_obj = RESIDUAL(bin_file,
                 fname_templates,
                 fname_spike_train,
                 n_batches,
                 len_recording,
                 sampling_rate,
                 n_channels,
                 fname_out,
                 dtype_in,
                 dtype_out)
        

residual_obj.compute_residual('residuals/')
residual_obj.save_residual()

### If residuals are already computed, input the location here
residual_file = 'residuals/residual.bin'


##### LOCALIZE SPIKES 

localizer_obj = LOCALIZER(bin_file, residual_file, dtype_input, fname_spike_train, fname_templates, geom_path, denoiser_weights, denoiser_min)
localizer_obj.get_offsets()
localizer_obj.compute_aligned_templates()
localizer_obj.load_denoiser()

for i in tqdm(range(n_batches)):
    localizer_obj.get_estimate(i, threshold = 6, output_directory = '/position_results_files')


##### Merge results 

input_directory = '/position_results_files'
output_directory = '/final_results'

get_merged_arrays(input_directory, output_directory, n_batches)

#### Load arrays and select spikes that have been localized 
#### This is necessary in order to visualize spikes
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

times_read = np.load(fname_times_read)
x_array = np.load(fname_x_merged)[times_read == 1]
z_array = np.load(fname_z_merged)[times_read == 1]
ptp_array = np.load(fname_max_ptp_merged)[times_read == 1]

np.save(fname_x_merged, x_array)
np.save(fname_max_ptp_merged, ptp_array)
np.save(fname_z_merged, z_array)


