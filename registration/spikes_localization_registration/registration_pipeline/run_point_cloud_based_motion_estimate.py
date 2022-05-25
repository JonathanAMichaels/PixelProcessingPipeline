import point_cloud_based_motion_estimate as pcbme
import numpy as np

### change paths to your localization files (i.e. x,y,z localizations)
x = 'x_results.npy'
y = 'y_results.npy'
z = 'z_results.npy'
times = 'spike_times.npy'
amps = 'max_ptp.npy'

### change path to geometry array
geomarray = np.load('geom.npy') # (num of channels, 2)

### set registration params
reg_win_num = 5 # set to 1 for rigid registration
iteration_num = 1 # supports 1 iteration for now

### length of the recording in seconds
T = 1000
subsampling_rate = np.log(T)/T

### get motion estimate
total_shift = pcbme.estimate_motion(x, y, z, times, amps, geomarray, subsampling_rate=subsampling_rate)

### save results
np.save('total_shift.npy', total_shift) # motion estimate
