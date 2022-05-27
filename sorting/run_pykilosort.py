import shutil
from pathlib import Path
import numpy as np

import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

def kilosort(config):
    SCRATCH_DIR = Path(config['neuropixel_folder'] + '/pykilosort')
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(exist_ok=True)
    DELETE = True  # delete the intermediate run products, if False they'll be copied over
    bin_file = Path(config['neuropixel'])
    # this is the output of the pykilosort data, unprocessed after the spike sorter
    ks_output_dir = Path(config['neuropixel_folder'] + '/results')
    ks_output_dir.mkdir(parents=True, exist_ok=True)

    params = ibl_pykilosort_params()
    print(params)
    run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR,
                          ks_output_dir=ks_output_dir, log_level='DEBUG', params=params)
