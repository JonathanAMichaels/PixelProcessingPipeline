import sys
import os
import glob
import scipy.io
from ruamel import yaml
from pathlib import Path
import datetime
import numpy as np
import shutil
from pipeline_utils import find, create_config, extract_sync, extract_LFP
from registration.registration import registration as registration_function
from sorting.pykilosort.run_myo_pykilosort import myo_sort as myo_function
from sorting.pykilosort.run_pykilosort import kilosort


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
myo_post = False
neuro_sorting = False
neuro_post = False
lfp_extract = False
cluster = False
in_cluster = False
if "-registration" in opts:
    registration = True
if "-myo_sorting" in opts:
    myo_sorting = True
if "-myo_post" in opts:
    myo_post = True
if "-neuro_sorting" in opts:
    neuro_sorting = True
if "-neuro_post" in opts:
    neuro_post = True
if "-lfp_extract" in opts:
    lfp_extract = True
if "-full" in opts:
    registration = True
    myo_sorting = True
    myo_post = True
    neuro_sorting = True
    neuro_post = True
if "-cluster" in opts:
    cluster = True
if "-in_cluster" in opts:
    in_cluster = True

if cluster:
    home = os.path.expanduser('~/')
    child_folder = Path(folder)
    child_folder = str(child_folder.stem)
    with open(home + 'scratch/slurm_job.sh', 'w') as f:
        f.write("#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=16\n" +
                "#SBATCH --mem=32G\n#SBATCH --time=1-00:00\n#SBATCH --account=def-andpru\n" +
                "module purge\nnvidia-smi\nsource ~/pipeline/bin/activate\n" +
                "scp -r " + folder + " $SLURM_TMPDIR/" + child_folder + "\n" +
                "module load gcc/9.3.0 arrow python/3.8.10 scipy-stack\n" +
                "python3 ~/PixelProcessingPipeline/pipeline.py -f $SLURM_TMPDIR/" + child_folder +
                " -registration -in_cluster"
                )
    os.chdir(home + 'scratch')
    os.system("sbatch slurm_job.sh")
    registration = False
    myo_sorting = False
    neuro_sorting = False

# Search working folder for existing configuration file
config_file = find('*.yaml', folder)
if len(config_file) > 1:
    raise SystemExit("There shouldn't be more than one config file in here (something went wrong)")
elif len(config_file) == 0:
    print('No config file found - creating one now')
    create_config(script_folder, folder)
    config_file = find('*.yaml', folder)
config_file = config_file[0]

# Load config
print('Using config file ' + config_file)
config = yaml.load(open(config_file, 'r'), Loader=yaml.RoundTripLoader)

# Check config for missing information and attempt to auto-fill
config['folder'] = folder
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
else:
    config['num_neuropixels'] = 0
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
temp = glob.glob(folder + '/*.kinarm')
if len(temp) == 0:
    print('No kinarm data in this recording session')
    config['kinarm'] = ''
else:
    config['kinarm'] = temp
if config['kinarm'] != '':
    print('Found kinarm data files')
config['script_dir'] = script_folder
if in_cluster:
    config['in_cluster'] = True
else:
    config['in_cluster'] = False

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
        if len(find('sync.mat', config_kilosort['neuropixel_folder'])) > 0:
            print('Found existing sync file')
        else:
            print('Extracting sync signal from ' + config_kilosort['neuropixel'] + ' and saving')
            extract_sync(config_kilosort)

        print('Starting spike sorting of ' + config_kilosort['neuropixel'])
        kilosort(config_kilosort)

# Proceed with neuro post-processing
if neuro_post:
    config_kilosort = {'script_dir': config['script_dir']}
    neuro_folders = glob.glob(config['neuropixel'] + '/*_g*')
    path_to_add = script_folder + '/sorting/'
    if os.path.isfile('/usr/local/MATLAB/R2021a/bin/matlab'):
        matlab_root = '/usr/local/MATLAB/R2021a/bin/matlab'  # something else for testing locally
    else:
        matlab_root = '/local/software/matlab/R2020b/bin/matlab'
    for pixel in range(config['num_neuropixels']):
        config_kilosort['neuropixel_folder'] = neuro_folders[pixel] + '/sorted'
        scipy.io.savemat('/tmp/config.mat', config_kilosort)
        os.system(matlab_root + ' -nodisplay -nosplash -nodesktop -r "addpath(genpath(\'' +
                  path_to_add + '\')); neuropixel_call"')

# Proceed with myo processing and spike sorting
if myo_sorting:
    config_kilosort = {'myomatrix': config['myomatrix'], 'script_dir': config['script_dir'],
                       'trange': np.array(config['Session']['trange']),
                       'sync_chan': int(config['Session']['myo_analog_chan'])}
    path_to_add = script_folder + '/sorting/'
    if os.path.isfile('/usr/local/MATLAB/R2021a/bin/matlab'):
        matlab_root = '/usr/local/MATLAB/R2021a/bin/matlab'  # something else for testing locally
    else:
        matlab_root = '/local/software/matlab/R2020b/bin/matlab'
    for myomatrix in range(len(config['Session']['myo_chan_list'])):
        f = glob.glob(config_kilosort['myomatrix'] + '/Record*')
        config_kilosort['myomatrix_data'] = f[0]
        config_kilosort['myomatrix_folder'] = config_kilosort['myomatrix'] + '/sorted' + str(myomatrix)
        config_kilosort['myomatrix_num'] = myomatrix
        config_kilosort['chans'] = np.array(config['Session']['myo_chan_list'][myomatrix])
        config_kilosort['num_chans'] = config['Session']['myo_chan_list'][myomatrix][1] - \
                                       config['Session']['myo_chan_list'][myomatrix][0] + 1

        scipy.io.savemat('/tmp/config.mat', config_kilosort)
        os.system(matlab_root + ' -nodisplay -nosplash -nodesktop -r "addpath(genpath(\'' +
                  path_to_add + '\')); myomatrix_binary"')

        shutil.rmtree(config_kilosort['myomatrix_folder'], ignore_errors=True)
        myo_function(config_kilosort)

# Proceed with myo post-processing
if myo_post:
    config_kilosort = {'script_dir': config['script_dir'], 'myomatrix': config['myomatrix']}
    path_to_add = script_folder + '/sorting/'
    if os.path.isfile('/usr/local/MATLAB/R2021a/bin/matlab'):
        matlab_root = '/usr/local/MATLAB/R2021a/bin/matlab'  # something else for testing locally
    else:
        matlab_root = '/local/software/matlab/R2020b/bin/matlab'
    for myomatrix in range(len(config['Session']['myo_chan_list'])):
        f = glob.glob(config_kilosort['myomatrix'] + '/Record*')

        config_kilosort['myomatrix_folder'] = config_kilosort['myomatrix'] + '/sorted' + str(myomatrix)
        config_kilosort['num_chans'] = config['Session']['myo_chan_list'][myomatrix][1] - \
                                       config['Session']['myo_chan_list'][myomatrix][0] + 1

        scipy.io.savemat('/tmp/config.mat', config_kilosort)

        print('Starting resorting of ' + config_kilosort['myomatrix_folder'])
        scipy.io.savemat('/tmp/config.mat', config_kilosort)
        os.system(matlab_root + ' -nodisplay -nosplash -nodesktop -r "addpath(genpath(\'' +
                  path_to_add + '\')); myomatrix_call"')

# Proceed with LFP extraction
if lfp_extract:
    config_kilosort['type'] = 1
    neuro_folders = glob.glob(config['neuropixel'] + '/*_g*')
    for pixel in range(config['num_neuropixels']):
        config_kilosort['neuropixel_folder'] = neuro_folders[pixel]
        tmp = glob.glob(neuro_folders[pixel] + '/*_t*.imec' + str(pixel) + '.ap.bin')
        config_kilosort['neuropixel'] = tmp[0]
        if len(find('lfp.mat', config_kilosort['neuropixel_folder'])) > 0:
            print('Found existing LFP file')
        else:
            print('Extracting LFP from ' + config_kilosort['neuropixel'] + ' and saving')
            extract_LFP(config_kilosort)


print('Pipeline finished! You\'ve earned a break.')
print(datetime.datetime.now())
