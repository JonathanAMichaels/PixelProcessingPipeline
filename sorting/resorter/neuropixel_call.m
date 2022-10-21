load('/tmp/config.mat')
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = false;
params.doPlots = false;
params.waveCount = 400;
params.backSp = 41;
params.forwardSp = 41;
params.corrRange = 10;
params.crit = 0.75;
params.consistencyThreshold = 0.7;

resorter(params)
quit;