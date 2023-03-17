#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["figure.figsize"] = (20, 12)


# In[2]:


dataset_folder = Path('/cifs/pruszynski/Malfoy/021623/021623_g0/021623_g0_imec0')
preprocess_folder = dataset_folder / 'preprocess'
corrected_folder = dataset_folder / 'corrected2'
peak_folder = dataset_folder / 'peaks'
peak_folder.mkdir(exist_ok=True)

spikeglx_folder = dataset_folder
# global kwargs for parallel computing
job_kwargs = dict(
    n_jobs=16,
    chunk_duration='2s',
    progress_bar=True,
)


# In[3]:


stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
raw_rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)
raw_rec
raw_rec.get_probe().to_dataframe()


# In[4]:


fig, ax = plt.subplots(figsize=(15, 10))
si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=True)
ax.set_ylim(-100, 100)


# In[5]:


from spikeinterface.preprocessing import highpass_spatial_filter
if not preprocess_folder.exists():
    rec_filtered = si.bandpass_filter(raw_rec, freq_min=300., freq_max=7000.)
    rec_shifted = si.phase_shift(rec_filtered)
    #bad_channel_ids = detect_bad_channels(rec_shifted)
    #rec_bad = interpolate_bad_channels(rec_shifted, bad_channel_ids)
    #rec_preprocessed = highpass_spatial_filter(rec_shifted)  
    rec_preprocessed = si.common_reference(rec_shifted, operator="median", reference="global")
    #rec_preprocessed.save(folder=preprocess_folder, **job_kwargs)
#rec_preprocessed = si.load_extractor(preprocess_folder)


# In[6]:


#rec_preprocessed = si.load_extractor(corrected_folder)


# In[7]:


# plot and check spikes
si.plot_timeseries(rec_preprocessed, time_range=(100, 110), channel_ids=raw_rec.channel_ids[40:70])


# In[6]:


noise_levels = si.get_noise_levels(rec_preprocessed, return_scaled=False)
noise_levels[noise_levels == 0] = 1
print(noise_levels)
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(noise_levels, bins=np.arange(0,10,1))
ax.set_title('noise across channel')


# In[7]:


from spikeinterface.sortingcomponents.peak_detection import detect_peaks
if not (peak_folder / 'peaks.npy').exists():
    peaks = detect_peaks(
        rec_preprocessed,
        method='locally_exclusive',
        local_radius_um=100,
        peak_sign='neg',
        detect_threshold=9, # 9 is good
        noise_levels=noise_levels,
        **job_kwargs,
    )
    np.save(peak_folder / 'peaks.npy', peaks)
peaks = np.load(peak_folder / 'peaks.npy')
print(peaks.shape)


# In[8]:


from spikeinterface.sortingcomponents.peak_selection import select_peaks
some_peaks = select_peaks(peaks, method='smart_sampling_amplitudes', noise_levels=noise_levels, n_peaks=500000)


# In[9]:


from spikeinterface.sortingcomponents.peak_localization import localize_peaks
all_kwargs = {**job_kwargs, **{'local_radius_um': 100., 'max_distance_um': 1000., 'optimizer': 'minimize_with_log_penality'}}
if not (peak_folder / 'peak_locations_monopolar_triangulation_log_limit.npy').exists():
    peak_locations = localize_peaks(
        rec_preprocessed,
        some_peaks,
        ms_before=0.3,
        ms_after=0.6,
        method='monopolar_triangulation',
        **all_kwargs
      )
    np.save(peak_folder / 'peak_locations_monopolar_triangulation_log_limit.npy', peak_locations)
    print(peak_locations.shape)


# In[10]:


from probeinterface.plotting import plot_probe
peak_locations = np.load(peak_folder / f'peak_locations_monopolar_triangulation_log_limit.npy')
probe = rec_preprocessed.get_probe()

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
ax = axs[0]
plot_probe(probe, ax=ax)
ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.002)
ax.set_xlabel('x')
ax.set_ylabel('y')
if 'z' in peak_locations.dtype.fields:
    ax = axs[1]
    ax.scatter(peak_locations['z'], peak_locations['y'], color='k', s=1, alpha=0.002)
    ax.set_xlabel('z')
ax.set_ylim(0, 4000)


# In[11]:


fig, ax = plt.subplots()
x = some_peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
y = peak_locations['y']
ax.scatter(x, y, s=1, color='k', alpha=0.05)
ax.set_ylim(0, 4000)


# In[69]:


from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
import torch
motion_kwargs = {'torch_device': torch.device('cuda'), 'conv_engine': 'torch',
                 'spatial_prior': True, 'corr_threshold': 0., 'time_horizon_s': 360, 'convergence_method': 'lsmr'}
motion, temporal_bins, spatial_bins, extra_check = estimate_motion(rec_preprocessed, some_peaks, peak_locations,
                    direction='y', win_step_um=50, win_sigma_um=600,
                    output_extra_check=True, progress_bar=True, **motion_kwargs)


# In[70]:


from spikeinterface.widgets import plot_pairwise_displacement, plot_displacement
plot_pairwise_displacement(motion, temporal_bins, spatial_bins, extra_check, ncols=8)


# In[71]:


#extra_check['spatial_hist_bins'] = extra_check['spatial_hist_bin_edges']
#extra_check['temporal_hist_bins'] = extra_check['temporal_hist_bin_edges']
plot_displacement(motion, temporal_bins, spatial_bins, extra_check, with_histogram=True)


# In[72]:


fig, ax = plt.subplots()
x = some_peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
y = peak_locations['y']
ax.scatter(x, y, s=1, color='k', alpha=0.05)
plot_displacement(motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=ax)
ax.set_ylim(-200, 4000)


# In[73]:


fig, ax = plt.subplots()
ax.plot(temporal_bins, motion)


# In[74]:


from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording
rec_corrected = CorrectMotionRecording(rec_preprocessed, motion, temporal_bins, spatial_bins,
                                            spatial_interpolation_method='idw',
                                            border_mode='remove_channels')


# In[76]:


import shutil
#shutil.rmtree(corrected_folder)
if not corrected_folder.exists():
    rec_corrected.save(folder=corrected_folder, format='binary', **job_kwargs)


# In[3]:


rec_corrected = si.load_extractor(corrected_folder)


# In[4]:


from spikeinterface.sorters import Kilosort2Sorter
kilosort_path = '/home/ROBARTS/jmichaels/PixelProcessingPipeline/sorting/Kilosort-2.0/'
Kilosort2Sorter.set_kilosort2_path(kilosort_path)
from spikeinterface.sorters import installed_sorters
installed_sorters()


# In[5]:


print('Starting spike sorting of ' + config_kilosort['neuropixel'])
    scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)
    os.system(matlab_root + ' -nodisplay -nosplash -nodesktop -r "addpath(\'' +
              path_to_add + '\'); Kilosort_run"')


# In[ ]:


# the results can be read back for futur session
sorting = si.read_sorter_folder(dataset_folder / 'kilosort2')
sorting


# In[ ]:


we = si.extract_waveforms(rec_corrected, sorting, folder=dataset_folder / 'kilosort2' / 'waveforms',
                          sparse=True, max_spikes_per_unit=500, ms_before=1.5,ms_after=2.,
                          **job_kwargs)


# In[ ]:


we = si.load_waveforms(dataset_folder / 'kilosort2' / 'waveforms')


# In[ ]:


metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                       'isi_violation', 'amplitude_cutoff'])
metrics

