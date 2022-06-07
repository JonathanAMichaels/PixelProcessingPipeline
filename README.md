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


for canada comp0ute

    module load gcc/9.3.0 arrow python scipy-stack
    virtualenv ~/pipeline
    source ~/pipeline/bin/activate
    pip install --upgrade pip setuptools wheel
    pip3 install torch torchvision
    pip install scipy ruamel.yaml ibl-neuropixel PyWavelets scikit-image pyfftw==0.12.0 cython pydantic
    pip install cupy




Install anaconda: https://docs.anaconda.com/anaconda/install/linux/

Create and setup the environment 

    conda env create -f setup.yml
    conda activate pipeline
    conda develop .

Compile codes necessary for drift correction

    cd registration/spike_localization_registration
    python3 setup.py build_ext --inplace
    pip install -e .

Ensure proper versions are installed

    conda install -c conda-forge cupy=10.5.0 cudatoolkit=10.2.89 "pyfftw=0.13.0=py39h51d1ae8_0"
 

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

