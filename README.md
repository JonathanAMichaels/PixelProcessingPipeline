# PixelProcessingPipeline
The purpose of this toolbox is to automatically process Neuropixels and Myomatrix data recorded during individual experiments

## Installation
These installation instructions were tested on the Computational Brain Science Group Server GPU image, and may need to be adjusted if run on another machine type.

The very first time you set up your virtual environment, follow these steps:
    virtualenv ~/pipeline
    source ~/pipeline/bin/activate
    pip install scipy ruamel.yaml ibllib PyWavelets scikit-image
    pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

Every time you start a new shell session, you will have to activate your virtual environement with
    source ~/pipeline/bin/activate
    
By default the incorrect version of nvcc is on the path. To fix that permanently, we need to modify our bashrc file
    nano ~/.bashrc
Scroll to bottom and add
    export PATH="/usr/local/cuda-11.2/bin:$PATH"
Save and close. Start a new shell.

To set up Kilosort the first time, open matlab
    module load matlab/R2021b
    ./srv/software/matlab/R2021b/bin/matlab
Navigate to the Kilosort CUDA directory wherever you installed this code and run mexGPUall
    cd('location_of_toolbox/sorting/Kilosort/CUDA')
    mexGPUAll
If all the files were built successfully, you are ready to go!

## Usage
Organize each experiment into one directory with a Neuropixel folder inside (e.g. 050322_g0), a Myomatrix folder (e.g. 2022-04-14_09-48-02_myo), which must have _myo at the end, and any .kinarm data files generated. You do not need to have all of these present.

The first time you process an experiment, call
    python3 pipeline.py -f /path_to_experiment -init

this will generate a config.yaml file in that directory with all the relevant parameters for that experiment generated automatically. Open that file with any text editor and add any session specific information to the Session parameter section. For example, if you collected myomatrix data you must specify which channels belong to which electrode and which channel is contains the sync information, since this information cannot be generated automatically.

If the config.yaml is correct, you can run the pipeline with the desired steps
    python3 pipeline.py -f /path_to_experiment -full
Alternatively, you can call any combination of
    -registration
    -neuro_sorting
    -myo_sorting
to perform only those steps.
