load('/tmp/config.mat')
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = false;
params.doPlots = false;
params.waveCount = 800;
params.backSp = 35;
params.forwardSp = 35;
params.corrRange = 20;
params.crit = 0.8;
params.consistencyThreshold = 0.7;

resorter(params)
quit;