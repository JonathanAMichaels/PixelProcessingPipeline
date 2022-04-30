import sys
import os
import glob
import scipy.io
from ruamel import yaml
from pipeline_utils import find, create_config, extract_sync
from registration.registration import registration as registration_function

script_folder = os.path.dirname(os.path.realpath(__file__))
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if "-f" in opts:
    folder = args[0]
    if os.path.isdir(folder):
        print('Using working folder ' + folder)
    else:
        raise SystemExit("Provided folder is not valid (you had one job...)")
else:
    raise SystemExit(f"Usage: {sys.argv[0]} -f argument must be present")
registration = False
myo_sorting = False
neuro_sorting = False
if "-registration" in opts:
    registration = True
if "-myo_sorting" in opts:
    myo_sorting = True
if "-neuro_sorting" in opts:
    neuro_sorting = True
if "-full" in opts:
    registration = True
    myo_sorting = True
    neuro_sorting = True
if "-init" in opts:
    registration = False
    myo_sorting = False
    neuro_sorting = False

# Search working folder for existing configuration file
config_file = find('*.yaml', folder)
if len(config_file) > 1:
    raise SystemExit("There shouldn't be two config files in here (something went wrong)")
elif len(config_file) == 0:
    print('No config file found - creating one now')
    create_config(script_folder, folder)
    config_file = find('*.yaml', folder)
config_file = config_file[0]

# Load config
print('Using config file ' + config_file)
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
    print('Using neuropixel folder ' + config['neuropixel'] + ' containing ' +
          str(config['num_neuropixels']) + ' neuropixel')
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
if config['myomatrix'] != '':
    print('Using myomatrix folder ' + config['myomatrix'])
if config['kinarm'] is None:
    temp = glob.glob(folder + '/*.kinarm')
    if len(temp) == 0:
        print('No kinarm data in this recording session')
        config['kinarm'] = ''
    else:
        config['kinarm'] = temp
if config['kinarm'] != '':
    print('Found kinarm data files')
config['script_dir'] = script_folder

# Save config file with up-to-date information
yaml.dump(config, open(config_file, 'w'), Dumper=yaml.RoundTripDumper)

# Proceed with registration
if registration:
    registration_function(config)

# Prepare common kilosort config
config_kilosort = yaml.safe_load(open(config_file, 'r'))
config_kilosort['myomatrix_number'] = 1
config_kilosort['channel_list'] = 1
# Proceed with neural spike sorting
if neuro_sorting:
    config_kilosort['type'] = 1
    neuro_folders = glob.glob(config['neuropixel'] + '/*_g*')
    for pixel in range(config['num_neuropixels']):
        config_kilosort['neuropixel_folder'] = neuro_folders[pixel]
        tmp = glob.glob(neuro_folders[pixel] + '/*_t*.imec' + str(pixel) + '.ap.bin')
        config_kilosort['neuropixel'] = tmp[0]
        scipy.io.savemat('/tmp/config.mat', config_kilosort)

        if len(find('sync.mat', config_kilosort['neuropixel_folder'])) > 0:
            print('Found existing sync file')
        else:
            print('Extracting sync signal from ' + config_kilosort['neuropixel'] + ' and saving')
            extract_sync(config_kilosort)

        print('Starting spike sorting of ' + config_kilosort['neuropixel'])
        path_to_add = script_folder + '/sorting'
        os.system('module load matlab/2021b')
        matlab_root = '/srv/software/matlab/R2021b/bin/matlab'
        #matlab_root = '/usr/local/MATLAB/R2021a/bin/matlab' # something else for testing locally
        os.system(matlab_root + ' -nodisplay -nosplash -nodesktop -r "addpath(genpath(\'' +
                  path_to_add + '\')); Kilosort_run"')

# Proceed with myo processing and spike sorting
if myo_sorting:
    config_kilosort = yaml.safe_load(open(config_file, 'r'))
    config_kilosort['type'] = 2
    config_kilosort['channel_list'] = range(16)
    config_kilosort['myomatrix_number'] = 1







