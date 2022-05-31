import os
import sys

script_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_folder)

import shutil
from pathlib import Path
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

def kilosort(config):
    SCRATCH_DIR = Path(config['neuropixel_folder'] + '/pykilosort')
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(exist_ok=True)
    DELETE = False  # delete the intermediate run products, if False they'll be copied over
    bin_file = Path(config['neuropixel_folder'] + '/NeuropixelsRegistration2/registered/standardized.bin')
    # this is the output of the pykilosort data, unprocessed after the spike sorter
    ks_output_dir = Path(config['neuropixel_folder'] + '/sorted')
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params([bin_file])
    params['perform_drift_registration'] = False
    #print(params)
    run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR,
                          ks_output_dir=ks_output_dir, log_level='INFO', params=params)
