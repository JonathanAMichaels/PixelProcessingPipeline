import shutil
from pathlib import Path
import numpy as np

import pykilosort
from pykilosort.ibl import run_spike_sorting_ibl, ibl_pykilosort_params

INTEGRATION_DATA_PATH = Path("/datadisk/Data/spike_sorting/pykilosort_tests")
SCRATCH_DIR = Path.home().joinpath("scratch", 'pykilosort')
shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
SCRATCH_DIR.mkdir(exist_ok=True)
DELETE = True  # delete the intermediate run products, if False they'll be copied over
bin_file = INTEGRATION_DATA_PATH.joinpath("imec_385_100s.ap.bin")
# this is the output of the pykilosort data, unprocessed after the spike sorter
ks_output_dir = INTEGRATION_DATA_PATH.joinpath("results")
ks_output_dir.mkdir(parents=True, exist_ok=True)
# this is the output standardized as per IBL standards (SI units, ALF convention)
alf_path = ks_output_dir.joinpath('alf')


params = ibl_pykilosort_params()
run_spike_sorting_ibl(bin_file, delete=DELETE, scratch_dir=SCRATCH_DIR,
                      ks_output_dir=ks_output_dir, alf_path=alf_path, log_level='DEBUG', params=params)
