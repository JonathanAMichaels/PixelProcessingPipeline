# PixelProcessingPipeline
This toolbox easily and automatically processes Neuropixels data recorded during experiments.

This toolbox will:
- Automatically generate a config file to keep track of experimental parameters
- For Neuropixel data:
  - Estimate the probe drift over time (Boussard et al. 2021, implemented in spikeinterface)
  - Perform drift registration using the above estimate sort with Kilosort 2.0 
  - Extract and save the sync signal sent from the behavioural task

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

CUDA Toolkit (Automatically installed with conda environment):
  - 11.3

### Instructions
Clone a copy of the repository on your local machine (for example, in the home directory)

    git clone https://github.com/JonathanAMichaels/PixelProcessingPipeline.git


#### Conda Environment 
If you **don't** already have a version of conda installed:

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
Organize each experiment into one directory with a Neuropixel folder inside (e.g. 041422_g0), and any kinarm files.

#### Conda Activation
Every time you open a new terminal, you must activate the environment.

    conda activate pipeline

#### Final Usage Steps
The first time you process an experiment, call

    python pipeline.py -f "/path/to/sessionYYYYMMDD"

This will generate a `config.yaml` file in that directory with all the relevant parameters for that experiment generated automatically. Open that file with any text editor and add any session specific information to the Session parameter section. For example, for Neuropixels data you need to specify the brain area of each electrode and the recording coordinates.

##### Spike Sorting Commands
To run a sort on the Neuropixel data, run:
    
    python pipeline.py -f "/path/to/sessionYYYYMMDD" -neuro_sort

##### Chaining Commands Together
If the `config.yaml` is correct, you can run the pipeline with both steps (registration and sorting), for example

    python pipeline.py -f "/path/to/sessionYYYYMMDD" -full

Alternatively, you can call any combination of:

    -registration
    -neuro_sort
    -lfp_extract

## Extensions
This code does not currently process .kinarm files or combine behavioural information with synced neural data. This may be added at a later date.