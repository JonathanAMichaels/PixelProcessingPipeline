# PixelProcessingPipeline
This toolbox easily and automatically processes Neuropixels and Myomatrix data recorded during experiments.

This toolbox will:
- Automatically generate a config file to keep track of experimental parameters
- For Neuropixel data:
  - Perform registration of data over time to correct for drift in the recording
  - Extract and save the sync signal sent from the behavioural task
  - Perform spike sorting with Kilosort 3
- For Myomatrix data:
  - Combine OpenEphys data into a single binary
  - Automatically remove broken channels
  - Extract and save the sync signal sent from the behavioural task
  - Create and save a 'bulk EMG' signal generated from the data
  - Perform spike sorting with Kilosort 3

## Installation

These installation instructions were tested on the Computational Brain Science Group Server 'CBS GPU 10GB' image, and may need to be adjusted if running on another machine type.

The very first time you set up your virtual environment, follow these steps:

Install anaconda: https://docs.anaconda.com/anaconda/install/linux/

Create and setup the environment 

    conda env create -f setup.yml
    conda activate pipeline
    conda develop .

Install pytorch

    pip3 install torch==1.9.0+cu112 torchvision==0.10.0+cu112 -f https://download.pytorch.org/whl/torch_stable.html

Compile codes necessary for drift correction

    cd registration/spike_localization_registration
    python3 setup.py build_ext --inplace
    pip install -e .

    conda install -c conda-forge cupy cudatoolkit=10.0
    conda install -c conda-forge "pyfftw=0.13.0=py39h51d1ae8_0"
 

## Usage

Organize each experiment into one directory with a Neuropixel folder inside (e.g. 041422_g0), a Myomatrix folder (e.g. 2022-04-14_09-48-02_myo, which must have _myo at the end) and any .kinarm data files generated. You do not need to have all of these present.

The first time you process an experiment, call

    python3 pipeline.py -f /path_to_experiment_folder -init

This will generate a config.yaml file in that directory with all the relevant parameters for that experiment generated automatically. Open that file with any text editor and add any session specific information to the Session parameter section. For example, if you collected Myomatrix data you must specify which channels belong to which electrode and which channel contains the sync information, since this information cannot be generated automatically.

If the config.yaml is correct, you can run the pipeline with all steps, for example

    python3 pipeline.py -f /cifs/pruszynski/Malfoy/041422 -full

Alternatively, you can call any combination of

    -registration
    -neuro_sorting
    -myo_sorting

to perform only those steps. Have fun!

## Extensions

This code does not currently process .kinarm files or combine behavioural information with synced neural data. This may be added at a later date.

The Neuropixels registration is based on https://github.com/evarol/NeuropixelsRegistration. While it works very well, I'm currently working on switching to an improved version.

