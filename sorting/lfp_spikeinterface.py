import spikeinterface.full as si
import numpy as np
from pathlib import Path
import shutil
import subprocess
from probeinterface import ProbeGroup
from probeinterface import write_prb, read_prb
from scipy.io import savemat
from sorting.readSGLX import readMeta

def unlock_files(directory):
    # Find the process IDs using the files in the directory
    process_ids = set()
    result = subprocess.run(['lsof', '+D', directory], stdout=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) > 2 and parts[1].isdigit():
            process_ids.add(parts[1])

    # Kill the processes
    for pid in process_ids:
        print(f"Killing process {pid}")
        subprocess.run(['kill', '-9', pid])

def lfp_extract(config):
    dataset_folder = Path(config['neuropixel_folder'])
    lfp_folder = dataset_folder / 'LFP'
    if lfp_folder.exists() and lfp_folder.is_dir():
        unlock_files(lfp_folder)
        shutil.rmtree(lfp_folder)

    spikeglx_folder = dataset_folder
    # global kwargs for parallel computing
    job_kwargs = dict(
        n_jobs=-1,
        chunk_duration='1s',
        progress_bar=True,
    )
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    print(stream_names)

    raw_rec = si.read_spikeglx(spikeglx_folder, stream_name=stream_names[0], load_sync_channel=False)

    P = raw_rec.get_probe()
    PRB = ProbeGroup()
    PRB.add_probe(P)

    params = {'LFP_filter_type': 'si.bandpass_filter', 'bandpass_frequency': (1, 300),
              'sampling_rate': 1000, 'gain': 100}

    rec1 = si.phase_shift(raw_rec)
    rec1 = si.bandpass_filter(recording=rec1, freq_min=params['bandpass_frequency'][0],
                              freq_max=params['bandpass_frequency'][1], dtype='float32')
    rec1 = si.resample(rec1, params['sampling_rate'])
    rec1 = si.scale(rec1, rec1.get_channel_gains())
    rec1 = si.scale(rec1, gain=params['gain'], dtype='int16')
    rec1.save(folder=lfp_folder, format='binary', **job_kwargs)
    savemat(lfp_folder / 'LFP_params.mat',
            {'electrode_contacts_um': PRB.to_dict()['probes'][0]['contact_positions'], 'params': params})
