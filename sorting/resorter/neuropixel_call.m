load('/tmp/config.mat')
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = false;
params.doPlots = false;
params.waveCount = 600;
params.backSp = 35;
params.forwardSp = 35;
params.corrRange = 10;
params.crit = 0.9;
params.consistencyThreshold = 0;
params.SNRThreshold = 2;
params.multiSNRThreshold = 2;
params.skipFilter = true;

resorter(params)
quit;