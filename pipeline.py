import concurrent.futures
import datetime
import glob
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
#from ibllib.ephys.spikes import ks2_to_alf
from ruamel.yaml import YAML

from pipeline_utils import create_config, extract_LFP, extract_sync, find
from registration.registration_spikeinterface import registration as registration_function
from sorting.sorting_spikeinterface import sorting as sorting_function
from sorting.Kilosort_gridsearch_config import get_KS_params_grid

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
registration_final = False
config = False
myo_config = False
myo_sort = False
myo_post = False
myo_plot = False
myo_phy = False
neuro_config = False
neuro_sort = False
neuro_post = False
lfp_extract = False
cluster = False
in_cluster = False
if "-registration" in opts:
    registration = True
if "-registration_final" in opts:
    registration_final = True
if "-config" in opts:
    config = True
if "-myo_config" in opts:
    myo_config = True
if "-myo_sort" in opts:
    myo_sort = True
if "-myo_post" in opts:
    myo_post = True
if "-myo_plot" in opts:
    myo_plot = True
if "-myo_phy" in opts:
    myo_phy = True
if "-neuro_config" in opts:
    neuro_config = True
if "-neuro_sort" in opts:
    neuro_sort = True
if "-neuro_post" in opts:
    neuro_post = True
if "-lfp_extract" in opts:
    lfp_extract = True
if "-full" in opts:
    registration = True
    myo_sort = True
    myo_post = True
    neuro_sort = True
    neuro_post = True
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
    myo_sort = False
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

temp_folder = glob.glob(folder + "/*_myo")
if len(temp_folder) > 1:
    SystemExit("There shouldn't be more than one Myomatrix folder")
elif len(temp_folder) == 0:
    print("No Myomatrix data in this recording session")
    config["myomatrix"] = ""
else:
    if os.path.isdir(temp_folder[0]):
        config["myomatrix"] = temp_folder[0]

# ensure global fields are present in config
if config["myomatrix"] != "":
    print("Using myomatrix folder " + config["myomatrix"])

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
GPU_str = ",".join([str(i) for i in config["GPU_to_use"]])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_str

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
config["registration_final"] = registration_final

# Save config file with up-to-date information
with open(config_file, "w") as f:
    yaml.dump(config, f)

# Proceed with registration
if registration or registration_final:
    registration_function(config)

# Prepare common kilosort config
with open(config_file) as f:
    config_kilosort = yaml.load(f)
config_kilosort["myomatrix_number"] = 1
config_kilosort["channel_list"] = 1

# if f"{config['script_dir']}/tmp" folder does not exist
if not os.path.isdir(f"{config['script_dir']}/tmp"):
    os.mkdir(f"{config['script_dir']}/tmp")

# Convenience function to edit neuro sorting config file
if neuro_config:
    if os.name == "posix":  # detect Unix
        subprocess.run(
            f"nano {config['script_dir']}/sorting/Kilosort_run.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-neuro_sort" done.')
        subprocess.run(
            f"nano {config['script_dir']}/sorting/resorter/neuropixel_call.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-neuro_post" done.')
    elif os.name == "nt":  # detect Windows
        subprocess.run(
            f"notepad {config['script_dir']}/sorting/Kilosort_run.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-neuro_sort" done.')

# Proceed with neural spike sorting
if neuro_sort:
    config_kilosort = {
        "script_dir": config["script_dir"],
        "trange": np.array(config["Session"]["trange"]),
    }
    config_kilosort["type"] = 1
    neuro_folders = glob.glob(config["neuropixel"] + "/*_g*")
    path_to_add = script_folder + "/sorting/"
    for pixel in range(config["num_neuropixels"]):
        config_kilosort["neuropixel_folder"] = neuro_folders[pixel]
        tmp = glob.glob(neuro_folders[pixel] + "/*_t*.imec" + str(pixel) + ".ap.bin")
        config_kilosort["neuropixel"] = tmp[0]
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

        print("Starting alf post-processing of " + config_kilosort["neuropixel"])
        alf_dir = Path(config_kilosort["neuropixel_folder"] + "/sorted/alf")
        shutil.rmtree(alf_dir, ignore_errors=True)
        #ks_dir = Path(config_kilosort["neuropixel_folder"] + "/sorted")
        #ks2_to_alf(ks_dir, Path(config_kilosort["neuropixel"]), alf_dir)

# Proceed with neuro post-processing
if neuro_post:
    config_kilosort = {"script_dir": config["script_dir"]}
    neuro_folders = glob.glob(config["neuropixel"] + "/*_g*")
    path_to_add = script_folder + "/sorting/"
    for pixel in range(config["num_neuropixels"]):
        config_kilosort["neuropixel_folder"] = (
            neuro_folders[pixel] + "/kilosort2/sorter_output"
        )
        scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)
        subprocess.run(
            [
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                f"addpath(genpath('{path_to_add}')); neuropixel_call",
            ],
            check=True,
        )

if myo_config:
    if os.name == "posix":  # detect Unix
        subprocess.run(
            f"nano {config['script_dir']}/sorting/Kilosort_run_myo_3_czuba.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-myo_sort" done.')
        subprocess.run(
            f"nano {config['script_dir']}/sorting/resorter/myomatrix_call.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-myo_post" done.')
    elif os.name == "nt":  # detect Windows
        subprocess.run(
            f"notepad {config['script_dir']}/sorting/Kilosort_run_myo_3_czuba.m",
            shell=True,
            check=True,
        )
        print('Configuration for "-myo_sort" done.')

# Proceed with myo processing and spike sorting
if myo_sort:
    config_kilosort = {
        "GPU_to_use": np.array(config["GPU_to_use"], dtype=int),
        "num_KS_jobs": int(config["num_KS_jobs"]),
        "myomatrix": config["myomatrix"],
        "script_dir": config["script_dir"],
        "recordings": np.array(config["recordings"], dtype=int)
        if type(config["recordings"][0]) != str
        else config["recordings"],
        "myo_data_passband": np.array(config["myo_data_passband"], dtype=float),
        "myo_data_sampling_rate": float(config["myo_data_sampling_rate"]),
        "num_KS_components": np.array(
            config["Sorting"]["num_KS_components"], dtype=int
        ),
        "trange": np.array(config["Session"]["trange"]),
        "sync_chan": int(config["Session"]["myo_analog_chan"]),
    }
    path_to_add = script_folder + "/sorting/"
    for myomatrix in range(len(config["Session"]["myo_chan_list"])):
        if config["concatenate_myo_data"]:
            config_kilosort["myomatrix_data"] = concatDataPath
        else:
            # find match to recording folder using recordings_str
            f = find("recording" + str(config["recordings"][0]), config["myomatrix"])
            config_kilosort["myomatrix_data"] = f[0]
            print(f"Using data from: {f[0]}")
        config_kilosort["myo_sorted_dir"] = (
            config_kilosort["myomatrix"] + "/sorted" + str(myomatrix)
        )
        config_kilosort["myomatrix_num"] = myomatrix
        config_kilosort["myo_chan_map_file"] = os.path.join(
            config["script_dir"],
            "geometries",
            config["Session"]["myo_chan_map_file"][myomatrix],
        )
        config_kilosort["chans"] = np.array(
            config["Session"]["myo_chan_list"][myomatrix]
        )
        config_kilosort["remove_bad_myo_chans"] = np.array(
            config["Session"]["remove_bad_myo_chans"][myomatrix]
        )
        config_kilosort["remove_channel_delays"] = np.array(
            config["Session"]["remove_channel_delays"][myomatrix]
        )
        config_kilosort["num_chans"] = (
            config["Session"]["myo_chan_list"][myomatrix][1]
            - config["Session"]["myo_chan_list"][myomatrix][0]
            + 1
        )
        scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)
        shutil.rmtree(config_kilosort["myo_sorted_dir"], ignore_errors=True)
        subprocess.run(
            [
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                (
                    "rehash toolboxcache; restoredefaultpath;"
                    f"addpath(genpath('{path_to_add}')); myomatrix_binary"
                ),
            ],
            check=True,
        )

        scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)

        # check if user wants to do grid search of KS params
        if config["Sorting"]["do_KS_param_gridsearch"] == 1:
            iParams = list(
                get_KS_params_grid()
            )  # get iterator of all possible param combinations
        else:
            # just pass an empty string to run once with chosen params
            iParams = [""]

        worker_ids = np.arange(config["num_KS_jobs"])
        # create new folders if running in parallel
        if config["num_KS_jobs"] > 1:
            # ensure proper configuration for parallel jobs
            assert config["num_KS_jobs"] <= len(
                config["GPU_to_use"]
            ), "Number of parallel jobs must be less than or equal to number of GPUs"
            assert (
                config["Sorting"]["do_KS_param_gridsearch"] == 1
            ), "Parallel jobs can only be used when do_KS_param_gridsearch is set to True"
            # create new folder for each parallel job to store results temporarily
            for i in worker_ids:
                # create new folder for each parallel job
                new_sorted_dir = config_kilosort["myo_sorted_dir"] + str(i)
                if os.path.isdir(new_sorted_dir):
                    shutil.rmtree(new_sorted_dir, ignore_errors=True)
                shutil.copytree(config_kilosort["myo_sorted_dir"], new_sorted_dir)
            # split iParams according to number of parallel jobs
            iParams_split = np.array_split(iParams, config["num_KS_jobs"])

        def run_KS_sorting(iParams, worker_id):
            iParams = iter(iParams)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config["GPU_to_use"][worker_id])
            save_path = (
                f"{config_kilosort['myo_sorted_dir']}{worker_id}"
                if config["num_KS_jobs"] > 1
                else config_kilosort["myo_sorted_dir"]
            )
            print(
                f"Starting spike sorting of {save_path} on GPU {config['GPU_to_use'][worker_id]}"
            )
            worker_id = str(worker_id)
            with tempfile.TemporaryDirectory(
                suffix=f"_worker{worker_id}"
            ) as worker_dir:
                while True:
                    # while no exhaustion of iterator
                    try:
                        these_params = next(iParams)
                        if type(these_params) == dict:
                            print(
                                f"Using these KS params from Kilosort_gridsearch_config.py"
                            )
                            print(these_params)
                            param_keys = list(these_params.keys())
                            param_keys_str = [f"'{k}'" for k in param_keys]
                            param_vals = list(these_params.values())
                            zipped_params = zip(param_keys_str, param_vals)
                            flattened_params = itertools.chain.from_iterable(
                                zipped_params
                            )
                            # this is a comma-separated string of key-value pairs
                            passable_params = ",".join(str(p) for p in flattened_params)
                        elif type(these_params) == str:
                            print(f"Using KS params from Kilosort_run_myo_3.m")
                            passable_params = (
                                these_params  # this is a string: 'default'
                            )
                        else:
                            print("ERROR: KS params must be a dictionary or a string.")
                            raise TypeError
                        if config["Sorting"]["do_KS_param_gridsearch"] == 1:
                            command_str = f"Kilosort_run_myo_3_czuba(struct({passable_params}),{worker_id},'{str(worker_dir)}');"
                        else:
                            command_str = f"Kilosort_run_myo_3_czuba('{passable_params}',{worker_id},'{str(worker_dir)}');"
                        subprocess.run(
                            [
                                "matlab",
                                "-nosplash",
                                "-nodesktop",
                                "-r",
                                (
                                    "rehash toolboxcache; restoredefaultpath;"
                                    f"addpath(genpath('{path_to_add}'));"
                                    f"{command_str}"
                                ),
                            ],
                            check=True,
                        )
                        # extract waveforms for Phy FeatureView
                        subprocess.run(
                            # "phy extract-waveforms params.py",
                            [
                                "phy",
                                "extract-waveforms",
                                "params.py",
                            ],
                            cwd=save_path,
                            check=True,
                        )
                        # get number of good units and total number of clusters from rez.mat
                        rez = scipy.io.loadmat(f"{save_path}/rez.mat")
                        num_KS_clusters = str(len(rez["good"]))
                        # sum the 1's in the good field of ops.mat to get number of good units
                        num_good_units = str(sum(rez["good"])[0])
                        brokenChan = scipy.io.loadmat(f"{save_path}/brokenChan.mat")[
                            "brokenChan"
                        ]
                        goodChans = np.setdiff1d(np.arange(1, 17), brokenChan)
                        goodChans_str = ",".join(str(i) for i in goodChans)

                        ## TEMP - remove this later: append git branch name to final_filename
                        # get git branch name
                        git_branches = subprocess.run(
                            ["git", "branch"], capture_output=True, text=True
                        )
                        git_branches = git_branches.stdout.split("\n")
                        git_branches = [i.strip() for i in git_branches]
                        git_branch = [i for i in git_branches if i.startswith("*")][0][
                            2:
                        ]

                        # remove spaces and single quoutes from passable_params string
                        time_stamp_us = datetime.datetime.now().strftime(
                            "%Y%m%d_%H%M%S%f"
                        )
                        filename_friendly_params = passable_params.replace(
                            "'", ""
                        ).replace(" ", "")
                        final_filename = (
                            f"sorted{str(myomatrix)}"
                            f"_{time_stamp_us}"
                            f"_rec-{recordings_str}"
                            # f"_chans-{goodChans_str}"
                            # f"_{num_good_units}-good-of-{num_KS_clusters}-total"
                            f"_{filename_friendly_params}"
                            # f"_{git_branch}"
                        )
                        # remove trailing underscore if present
                        final_filename = (
                            final_filename[:-1]
                            if final_filename.endswith("_")
                            else final_filename
                        )
                        # store final_filename in a new ops.mat field in the sorted0 folder
                        ops = scipy.io.loadmat(f"{save_path}/ops.mat")
                        ops.update({"final_myo_sorted_dir": final_filename})
                        scipy.io.savemat(f"{save_path}/ops.mat", ops)

                        # copy sorted0 folder tree to a new folder with timestamp to label results by params
                        # this serves as a backup of the sorted0 data, so it can be loaded into Phy later
                        shutil.copytree(
                            save_path,
                            Path(save_path).parent.joinpath(final_filename),
                        )

                    except StopIteration:
                        if config["Sorting"]["do_KS_param_gridsearch"] == 1:
                            print(f"Grid search complete for worker {worker_id}")
                        return  # exit the function
                    except:
                        if config["Sorting"]["do_KS_param_gridsearch"] == 1:
                            print("Error in grid search.")
                        else:
                            print("Error in sorting.")
                        raise  # re-raise the exception

        if config["num_KS_jobs"] > 1:
            # run parallel jobs
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(run_KS_sorting, iParams_split, worker_ids)
        else:
            # run single job
            run_KS_sorting(iParams, worker_ids[0])

# Proceed with myo post-processing
if myo_post:
    config_kilosort = {
        "script_dir": config["script_dir"],
        "myomatrix": config["myomatrix"],
        "GPU_to_use": config["GPU_to_use"],
    }
    path_to_add = script_folder + "/sorting/"
    for myomatrix in range(len(config["Session"]["myo_chan_list"])):
        f = glob.glob(config_kilosort["myomatrix"] + "/Record*")

        config_kilosort["myo_sorted_dir"] = (
            (config_kilosort["myomatrix"] + "/sorted" + str(myomatrix))
            if "-d" not in opts
            else (config_kilosort["myomatrix"] + "/" + previous_sort_folder_to_use)
        )
        config_kilosort["myo_chan_map_file"] = os.path.join(
            config["script_dir"],
            "geometries",
            config["Session"]["myo_chan_map_file"][myomatrix],
        )
        config_kilosort["remove_bad_myo_chans"] = np.array(
            config["Session"]["remove_bad_myo_chans"][myomatrix]
        )
        config_kilosort["remove_channel_delays"] = np.array(
            config["Session"]["remove_channel_delays"][myomatrix]
        )
        config_kilosort["num_chans"] = (
            config["Session"]["myo_chan_list"][myomatrix][1]
            - config["Session"]["myo_chan_list"][myomatrix][0]
            + 1
        )

        scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)
        shutil.rmtree(config_kilosort["myo_sorted_dir"] + "/Plots", ignore_errors=True)

        print("Starting resorting of " + config_kilosort["myo_sorted_dir"])
        scipy.io.savemat(f"{config['script_dir']}/tmp/config.mat", config_kilosort)
        ## get intermediate merge folders -- (2023-09-11) not doing intermediate merges anymore
        # merge_folders = Path(f"{config_kilosort['myo_sorted_dir']}/custom_merges").glob(
        #     "intermediate_merge*"
        # )
        subprocess.run(
            [
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                (
                    "rehash toolboxcache; restoredefaultpath;"
                    f"addpath(genpath('{path_to_add}')); myomatrix_call"
                ),
            ],
            check=True,
        )

        # # extract waveforms for Phy FeatureView
        # for iDir in merge_folders:
        #     # create symlinks to processed data
        #     Path(f"{iDir}/proc.dat").symlink_to(Path("../../proc.dat"))
        #     # run Phy extract-waveforms on intermediate merges
        #     subprocess.run(["phy", "extract-waveforms", "params.py"], cwd=iDir, check=True)
        # create symlinks to processed data
        Path(
            f"{config_kilosort['myo_sorted_dir']}/custom_merges/final_merge/proc.dat"
        ).symlink_to(Path("../../proc.dat"))
        # run Phy extract-waveforms on final merge
        subprocess.run(
            ["phy", "extract-waveforms", "params.py"],
            cwd=f"{config_kilosort['myo_sorted_dir']}/custom_merges/final_merge",
            check=True,
        )

        # copy sorted0 folder tree into same folder as for -myo_sort
        try:
            merge_path = "custom_merges/final_merge"
            shutil.copytree(
                Path(config_kilosort["myo_sorted_dir"]).joinpath(merge_path),
                Path(config_kilosort["myo_sorted_dir"])
                .parent.joinpath(previous_sort_folder_to_use)
                .joinpath(merge_path),
            )
        except FileExistsError:
            print(
                f"WARNING: Final merge already exists in {previous_sort_folder_to_use}, files not updated"
            )
        except:
            raise

# plot to show spikes overlaid on electrophysiology data, for validation purposes
if myo_plot:
    path_to_add = script_folder + "/sorting/"
    if "-d" in opts:
        sorted_folder_to_plot = previous_sort_folder_to_use
        args = args[1:]  # remove the -d flag related argument
    # create default values for validation plot arguments, if not provided
    if len(args) == 1:
        arg1 = int(1)  # default to plot chunk 1
        arg2 = "true"  # default to logical true to show all clusters
    elif len(args) == 2:
        arg1 = int(args[1])
        arg2 = "true"  # default to logical true to show all clusters
    elif len(args) == 3:
        import json

        arg_as_list = json.loads(args[2])
        arg1 = int(args[1])
        arg2 = np.array(arg_as_list).astype(int)
    subprocess.run(
        [
            "matlab",
            "-nodesktop",
            "-nosplash",
            "-r",
            (
                "rehash toolboxcache; restoredefaultpath;"
                f"addpath(genpath('{path_to_add}')); spike_validation_plot({arg1},{arg2})"
            ),
        ],
        check=True,
    )

if myo_phy:
    path_to_add = script_folder + "/sorting/"
    if "-d" in opts:
        sorted_folder_to_plot = previous_sort_folder_to_use
        args = args[1:]  # remove the -d flag related argument
    else:
        # default to sorted0 folder, may need to update to be flexible for sorted1, 2, etc.
        sorted_folder_to_plot = "sorted0"
    os.chdir(Path(config["myomatrix"]).joinpath(sorted_folder_to_plot))
    subprocess.run(
        [
            "phy",
            "template-gui",
            "params.py",
        ],
    )

# Proceed with LFP extraction
if lfp_extract:
    config_kilosort["type"] = 1
    neuro_folders = glob.glob(f"{config['neuropixel']}'/*_g*")
    for pixel in range(config["num_neuropixels"]):
        config_kilosort["neuropixel_folder"] = neuro_folders[pixel]
        tmp = glob.glob(f"{neuro_folders[pixel]}/*_t*.imec{str(pixel)}.ap.bin")
        config_kilosort["neuropixel"] = tmp[0]
        if len(find("lfp.mat", config_kilosort["neuropixel_folder"])) > 0:
            print("Found existing LFP file")
        else:
            print(f"Extracting LFP from {config_kilosort['neuropixel']} and saving")
            extract_LFP(config_kilosort)


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
