
    rec_corrected = si.load_extractor(corrected_folder)



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
