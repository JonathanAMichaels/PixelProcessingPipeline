import os
import sys
import glob
import h5py

script_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_folder)

import shutil
from pathlib import Path
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

def kilosort(config):
    bin_file = Path(config['neuropixel'])
    ks_output_dir = Path(config['neuropixel_folder'] + '/sorted')
    scratch_dir = ks_output_dir
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params([bin_file])
    #params['perform_drift_registration'] = False

    registered_file = glob.glob(config['neuropixel_folder'] + '/NeuropixelsRegistration2/' + 'subtraction_*.h5')
    with h5py.File(registered_file[0], "r") as f:
        dispmap = f["dispmap"][:]

    params['nblocks'] = dispmap.shape[0]
    params['disp_map'] = dispmap.tolist()
    run_spike_sorting_ibl(bin_file, delete=True, scratch_dir=scratch_dir,
                          ks_output_dir=ks_output_dir, log_level='INFO', params=params)
