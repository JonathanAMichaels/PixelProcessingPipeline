# PixelProcessingPipeline
This toolbox easily and automatically processes Neuropixels data using spikeinterface

This toolbox will:
- Automatically generate a config file to keep track of experimental parameters
- Extract and save the sync signal sent from the behavioural task
- For Neuropixels data there are two routes:
  1)
    - Estimate the probe drift over time (Boussard et al. 2021, implemented in spikeinterface)
    - Perform drift registration and spike sort with Kilosort 2.0
  2)
    - Estimate and perform drift with Kilosort 4.0 and spike sort with Kilosort 4.0


## Installation
### Requirements

Many processing steps require a CUDA capable GPU.
  - For Neuropixel data, a GPU with at least 10GB of onboard RAM is recommended

### Instructions
Clone a copy of the repository on your local machine (for example, in the home directory)

    git clone https://github.com/JonathanAMichaels/PixelProcessingPipeline.git

## Installation
Install spikeinterface from https://github.com/SpikeInterface/spikeinterface

Install Kilosort 4 from https://github.com/MouseLand/Kilosort

## Usage
Organize each experiment into one directory with a Neuropixel folder inside (e.g. 041422_g0), and any kinarm files.

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

    -neuro_sort
    -lfp_extract

## Extensions
This code does not currently process .kinarm files or combine behavioural information with synced neural data. This may be added at a later date.