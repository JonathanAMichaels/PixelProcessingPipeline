# PixelProcessingPipeline
This toolbox easily and automatically processes Neuropixels and Myomatrix data recorded during experiments.

This toolbox will:
- Automatically generate a config file to keep track of experimental parameters
- For Neuropixel data:
  - Estimate the probe drift over time (Boussard et al. 2021)
  - Perform drift registration using the above estimate with Kilosort 2.5 (ibl pykilosort) and sorting with Kilosort 2.0 
  - Extract and save the sync signal sent from the behavioural task
- For Myomatrix data:
  - Combine OpenEphys data into a single binary and automatically remove broken channels
  - Extract and save the sync signal sent from the behavioural task
  - Perform spike sorting with a modified version of Kilosort 3.0
  - Combine similar units, calculate motor unit statistics, export back to phy

## Folder Tree Structure
![Alt text](images/folder_tree_structure.png)

## Installation
### Requirements
Currently, using a Linux-based OS is recommended. The code has been tested on Ubuntu and CentOS. Windows support is experimental and may require additional changes.

Many processing steps require a CUDA capable GPU.
  - For Neuropixel data, a GPU with at least 10GB of onboard RAM is recommended

Required MATLAB Toolboxes:
  - Parallel Computing Toolbox
  - Signal Processing Toolbox
  - Statistics and Machine Learning Toolbox

Nvidia Driver:
  - Linux:      >=450.80.02
  - Windows:    >=452.39

CUDA Toolkit (Automatically installed with micromamba/conda environment):
  - 11.3

### Instructions
Clone a copy of the repository on your local machine (for example, in the home directory)

    git clone https://github.com/JonathanAMichaels/PixelProcessingPipeline.git


#### Conda Environment
ss

    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh

To install using a conda environment, follow these steps:


    conda env create -f environment.yml
    conda activate pipeline
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


#### Final Installation Steps
Open matlab and confirm that all mex files compile by running
    
    WARNING: make sure to activate the pipeline environment before running these commands

    matlab -nodesktop
    cd PixelProcessingPipeline/sorting/Kilosort-2.0/CUDA/
    mexGPUall


## Usage
Organize each experiment into one directory with a Neuropixel folder inside (e.g. 041422_g0), a Myomatrix folder (e.g. 2022-04-14_09-48-02_myo, which must have _myo at the end) and any .kinarm data files generated.
The Myomatrix folder must be organized either as 'folder_myo/Record Node ###/continuous/' for binary open ephys data, or as 'folder_myo/Record Node ###/***.continuous' for open ephys format data.

Each time a sort is performed, a new folder will be created in the experiment directory with the date and time of the sort. Inside this folder will be the sorted data, the phy output files, and a copy of the ops used to sort the data. The original OpenEphys data will not be modified.

#### Conda Activation
Every time you open a new terminal, you must activate the environment.

    conda activate pipeline

#### Final Usage Steps
The first time you process an experiment, call

    python pipeline.py -f "/path/to/sessionYYYYMMDD"

This will generate a `config.yaml` file in that directory with all the relevant parameters for that experiment generated automatically. Open that file with any text editor and add any session specific information to the Session parameter section. For example, if you collected Myomatrix data you must specify which channels belong to which electrode and which channel contains the sync information, since this information cannot be generated automatically.

##### Configuration Commands
Editing the main configuration file can be done by running the command below:
    
    python pipeline.py -f "/path/to/sessionYYYYMMDD" -config

To edit the configuration file for the processing Neuropixel data, run

    python pipeline.py -f "/path/to/sessionYYYYMMDD" -neuro_config

##### Spike Sorting Commands
To run a sort on the Neuropixel data, run
    
    python pipeline.py -f "/path/to/sessionYYYYMMDD" -neuro_sort

##### Chaining Commands Together
If the `config.yaml` is correct, you can run the pipeline with all steps, for example

    python pipeline.py -f "/path/to/sessionYYYYMMDD" -full

Alternatively, you can call any combination of:

    -config
    -registration
    -neuro_config
    -neuro_sort
    -lfp_extract

to perform only those steps. For example, if you want to configure and immediately spike sort, run

    python pipeline.py -f "/path/to/sessionYYYYMMDD" -config -neuro_sort

## Extensions
This code does not currently process .kinarm files or combine behavioural information with synced neural data. This may be added at a later date.