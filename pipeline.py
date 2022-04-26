import sys
import os
import yaml
from pipeline_utils import find, create_config

script_folder = os.getcwd()
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
    SystemExit("There shouldn't be two config files in here (something went wrong)")
elif len(config_file) == 0:
    create_config(script_folder, folder)
    config_file = find('*.yaml', folder)
config_file = config_file[0]

# Load config
print(config_file)
with open(config_file) as f:
    config = yaml.load(f)
    print(config)







