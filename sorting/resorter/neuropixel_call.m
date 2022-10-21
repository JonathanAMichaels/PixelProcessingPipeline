load('/tmp/config.mat')
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = false;
params.doPlots = false;
params.waveCount = 1000;
params.backSp = 51;
params.forwardSp = 51;
params.corrRange = 10;
params.crit = 0.75;
params.consistencyThreshold = 0.7;

resorter(params)
quit;