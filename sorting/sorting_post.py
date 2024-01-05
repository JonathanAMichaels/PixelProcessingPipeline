import spikeinterface.full as si
from pathlib import Path

def sorting_post(config):
    dataset_folder = Path(config['neuropixel_folder'])
    sorting_folder = dataset_folder / 'kilosort2.0'
    waveform_folder = sorting_folder / 'waveforms_kilosort2.0'
    we = si.WaveformExtractor.load(waveform_folder)
    metrics = si.compute_quality_metrics(we, metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation',
                                                           'amplitude_median', 'amplitude_cutoff'])
    metrics.to_csv(sorting_folder / 'metrics.csv')
