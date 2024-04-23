import datetime
import glob
import os
import subprocess
import sys
from pathlib import Path
from ruamel.yaml import YAML
from pipeline_utils import create_config, extract_sync, find
from registration.registration_spikeinterface import registration as registration_function
from sorting.sorting_kilosort import sorting as sorting_function
from sorting.lfp_spikeinterface import lfp_extract as lfp_function
from sorting.sorting_post import sorting_post as sorting_post_function

# calculate time taken to run each pipeline call
start_time = datetime.datetime.now()


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


script_folder = os.path.dirname(os.path.realpath(__file__))
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# use -f option to specify working folder for this session
if "-f" in opts:
    folder = args[0]
    if os.path.isdir(folder):
        print("Using working folder " + folder)
    else:
        raise SystemExit("Provided folder is not valid (you had one job...)")
else:
    raise SystemExit(
        f"Usage: {sys.argv[0]} -f argument must be present. Also, ensure environment is activated."
    )

registration = False
config = False
neuro_config = False
neuro_sort = False
lfp_extract = False
cluster = False
in_cluster = False
if "-registration" in opts:
    registration = True
if "-config" in opts:
    config = True
if "-neuro_sort" in opts:
    neuro_sort = True
if "-lfp_extract" in opts:
    lfp_extract = True
if "-full" in opts:
    registration = True
    neuro_sort = True
if "-cluster" in opts:
    cluster = True
if "-in_cluster" in opts:
    in_cluster = True

if cluster:
    home = os.path.expanduser("~/")
    child_folder = Path(folder)
    child_folder = str(child_folder.stem)
    with open(home + "scratch/slurm_job.sh", "w") as f:
        f.write(
            "#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --cpus-per-task=16\n"
            + "#SBATCH --mem=32G\n#SBATCH --time=1-00:00\n#SBATCH --account=def-andpru\n"
            + "module purge\nnvidia-smi\nsource ~/pipeline/bin/activate\n"
            + "scp -r "
            + folder
            + " $SLURM_TMPDIR/"
            + child_folder
            + "\n"
            + "module load gcc/9.3.0 arrow python/3.8.10 scipy-stack\n"
            + "python3 ~/PixelProcessingPipeline/pipeline.py -f $SLURM_TMPDIR/"
            + child_folder
            + " -registration -in_cluster"
        )
    os.chdir(home + "scratch")
    os.system("sbatch slurm_job.sh")
    registration = False
    neuro_sort = False

# Search working folder for existing configuration file
config_file = find("config.yaml", folder)
if len(config_file) > 1:
    raise SystemExit(
        "There shouldn't be more than one config file in here (something went wrong)"
    )
elif len(config_file) == 0:
    print("No config file found - creating one now")
    create_config(script_folder, folder)
    config_file = find("config.yaml", folder)
config_file = config_file[0]

if config:
    if os.name == "posix":  # detect Unix
        subprocess.run(f"nano {config_file}", shell=True, check=True)
        print("Configuration done.")
    elif os.name == "nt":  # detect Windows
        subprocess.run(f"notepad {config_file}", shell=True, check=True)
        print("Configuration done.")

# Load config
print("Using config file " + config_file)
# make round-trip loader
yaml = YAML()
with open(config_file) as f:
    config = yaml.load(f)

# Check config for missing information and attempt to auto-fill
config["folder"] = folder
temp_folder = glob.glob(folder + "/*_g0")
if len(temp_folder) > 1:
    raise SystemExit("There shouldn't be more than one Neuropixel folder")
elif len(temp_folder) == 0:
    print("No Neuropixel data in this recording session")
    config["neuropixel"] = ""
else:
    if os.path.isdir(temp_folder[0]):
        config["neuropixel"] = temp_folder[0]
    else:
        raise SystemExit("Provided folder is not valid")
if config["neuropixel"] != "":
    temp_folder = glob.glob(config["neuropixel"] + "/" + "*_g*")
    config["num_neuropixels"] = len(temp_folder)
    print(
        "Using neuropixel folder "
        + config["neuropixel"]
        + " containing "
        + str(config["num_neuropixels"])
        + " neuropixel"
    )
else:
    config["num_neuropixels"] = 0

assert (
    config["num_neuropixels"] >= 0
), "Number of neuropixels must be greater than or equal to 0"

# find MATLAB installation
if os.path.isfile("/usr/local/MATLAB/R2021a/bin/matlab"):
    matlab_root = (
        "/usr/local/MATLAB/R2021a/bin/matlab"  # something else for testing locally
    )
elif os.path.isfile("/srv/software/matlab/R2021b/bin/matlab"):
    matlab_root = "/srv/software/matlab/R2021b/bin/matlab"
else:
    matlab_path = glob.glob("/usr/local/MATLAB/R*")
    matlab_root = matlab_path[0] + "/bin/matlab"


# set chosen GPUs in environment variable
#GPU_str = ",".join([str(i) for i in config["GPU_to_use"]])
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = GPU_str

temp = glob.glob(folder + "/*.kinarm")
if len(temp) == 0:
    print("No kinarm data in this recording session")
    config["kinarm"] = ""
else:
    config["kinarm"] = temp
if config["kinarm"] != "":
    print("Found kinarm data files")
config["script_dir"] = script_folder
if in_cluster:
    config["in_cluster"] = True
else:
    config["in_cluster"] = False

# Save config file with up-to-date information
with open(config_file, "w") as f:
    yaml.dump(config, f)

# Proceed with registration
if registration:
    registration_function(config)

# if f"{config['script_dir']}/tmp" folder does not exist
if not os.path.isdir(f"{config['script_dir']}/tmp"):
    os.mkdir(f"{config['script_dir']}/tmp")

# Proceed with neural spike sorting
if neuro_sort:
    neuro_folders = glob.glob(config["neuropixel"] + "/*_g*")
    for pixel in range(config["num_neuropixels"]):
        tmp = glob.glob(neuro_folders[pixel] + "/*_t*.imec" + str(pixel) + ".ap.bin")
        config_kilosort = {"neuropixel_folder": neuro_folders[pixel], "neuropixel": tmp[0]}
        if len(find("sync.mat", config_kilosort["neuropixel_folder"])) > 0:
            print("Found existing sync file")
        else:
            print(
                "Extracting sync signal from "
                + config_kilosort["neuropixel"]
                + " and saving"
            )
            extract_sync(config_kilosort)

        print("Starting spike sorting of " + config_kilosort["neuropixel"])
        sorting_function(config_kilosort)

# Proceed with LFP extraction
if lfp_extract:
    neuro_folders = glob.glob(config["neuropixel"] + "/*_g*")
    for pixel in range(config["num_neuropixels"]):
        tmp = glob.glob(neuro_folders[pixel] + "/*_t*.imec" + str(pixel) + ".ap.bin")
        config_kilosort = {"neuropixel_folder": neuro_folders[pixel], "neuropixel": tmp[0]}
        print(f"Extracting LFP from {config_kilosort['neuropixel']} and saving")
        lfp_function(config_kilosort)


print("Pipeline finished! You've earned a break.")
finish_time = datetime.datetime.now()
time_elapsed = finish_time - start_time
# use strfdelta to format time elapsed
print(
    (
        "Time elapsed: "
        f"{strfdelta(time_elapsed, '{hours} hours, {minutes} minutes, {seconds} seconds')}"
    )
)

# reset the terminal mode to prevent not printing user input to terminal after program exits
subprocess.run(["stty", "sane"])
