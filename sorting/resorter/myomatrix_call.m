script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))
load(fullfile(script_dir, '/tmp/ops.mat'))
load(myo_chan_map_file)

% resorting parameters
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = [myo_sorted_dir '/custom_merges'];
params.binaryFile = [myo_sorted_dir '/data.bin'];
params.doPlots = true; % whether to generate plots
params.savePlots = true; % whether to save plots
params.skipFilter = false;
params.SNRThresh = 2.0;
params.corrThresh = 0.9; % minimum correlation to be considered as originating from one cluster
params.consistencyThresh = 0.6; % minimum consistency to be considered as originating from one cluster
params.spikeCountLim = 10; % minimum spike count to be included in output
params.refractoryLim = 1; % spikes below this refractory time limit will be considered duplicates

% make sure a sorting exists
if isfile([myo_sorted_dir '/spike_times.npy'])
    resorter(params)
else
    disp('No spike sorting to post-process')
end

quit;
