script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])

% resorting parameters
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/recording.dat'];
params.userSorted = false;
params.doPlots = false;
params.savePlots = false;
params.waveCount = 600;
params.backSp = 35;
params.forwardSp = 35;
params.corrRange = 20;
params.corrThresh = 0.9;
params.consistencyThresh = 0.6;
params.skipFilter = true;

resorter(params)
quit;
