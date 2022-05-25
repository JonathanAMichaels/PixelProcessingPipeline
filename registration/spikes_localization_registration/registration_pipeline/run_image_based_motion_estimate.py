import image_based_motion_estimate as ibme
import numpy as np

### change paths to your localization files (e.g. z localizations), and specify the direction
locs = 'z_results.npy'
times = 'spike_times.npy'
amps = 'max_ptp.npy'
direction = 'z'

### change path to geometry array
geomarray = np.load('geom.npy') # (num of channels, 2)

### set registration params
reg_win_num = 10 # set to 1 for rigid registration
reg_block_num = 100 # set to 1 for rigid registration
iteration_num = 2

# length of the recording in seconds
T = 1000
subsampling_rate = np.log(T)/T

### get motion estimate
total_shift = ibme.estimate_motion(locs, times, amps, 
        geomarray, direction, subsampling_rate=subsampling_rate)

### save results
np.save('total_shift.npy', total_shift) # motion estimate

### registered raster plots will be saved in ./image_based_registered_raster
