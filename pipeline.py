import sys
import os, glob
from ruamel import yaml
from pipeline_utils import find, create_config
from registration.registration import registration as registration_function

script_folder = os.path.dirname(os.path.realpath(__file__))
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if "-f" in opts:
    folder = args[0]
    if os.path.isdir(folder):
        print('Working folder: ' + folder)
    else:
        raise SystemExit("Provided folder is not valid (you had one job...)")
else:
    raise SystemExit(f"Usage: {sys.argv[0]} -f argument must be present")
registration = False
myo_sorting = False
neuro_sorting = False
myo_preprocess = False
if "-registration" in opts:
    registration = True
if "-myo_sorting" in opts:
    myo_sorting = True
if "-neuro_sorting" in opts:
    neuro_sorting = True
if "-myo_preprocess" in opts:
    myo_preprocess = True
if "-full" in opts:
    registration = True
    myo_sorting = True
    neuro_sorting = True
    myo_preprocess = True

# Search working folder for existing configuration file
config_file = find('*.yaml', folder)
if len(config_file) > 1:
    raise SystemExit("There shouldn't be two config files in here (something went wrong)")
elif len(config_file) == 0:
    create_config(script_folder, folder)
    config_file = find('*.yaml', folder)
config_file = config_file[0]

# Load config
config = yaml.load(open(config_file, 'r'), Loader=yaml.RoundTripLoader)

# Check config for missing information and attempt to auto-fill
if config['folder'] is None:
    config['folder'] = folder
if config['neuropixel'] is None:
    temp_folder = glob.glob(folder + "/*_g0")
    if len(temp_folder) > 1:
        raise SystemExit("There shouldn't be more than one Neuropixel folder")
    elif len(temp_folder) == 0:
        print('No Neuropixel data in this recording session')
        config['neuropixel'] = ''
    else:
        if os.path.isdir(temp_folder[0]):
            config['neuropixel'] = temp_folder[0]
        else:
            raise SystemExit("Provided folder is not valid")
if config['neuropixel'] != '':
    temp_folder = glob.glob(config['neuropixel'] + '/' + '*_g*')
    config['num_neuropixels'] = len(temp_folder)
if config['myomatrix'] is None:
    temp_folder = glob.glob(folder + '/*_myo')
    if len(temp_folder) > 1:
        SystemExit("There shouldn't be more than one Myomatrix folder")
    elif len(temp_folder) == 0:
        print('No Myomatrix data in this recording session')
        config['myomatrix'] = ''
    else:
        if os.path.isdir(temp_folder[0]):
            config['myomatrix'] = temp_folder[0]
if config['kinarm'] is None:
    temp = glob.glob(folder + '/*.kinarm')
    if len(temp) == 0:
        print('No kinarm data in this recording session')
        config['kinarm'] = ''
    else:
        config['kinarm'] = temp
if config['script_dir'] is None:
    config['script_dir'] = script_folder

# Save config file with up-to-date information
yaml.dump(config, open(config_file, 'w'), Dumper=yaml.RoundTripDumper)

# Proceed with registration
if registration:
    registration_function(config)







