load('/tmp/config.mat')
load([script_dir '/geometries/neuropixPhase3B1_kilosortChanMap'])
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = neuropixel_folder;
params.binaryFile = [neuropixel_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = false;
params.doPlots = false;
params.temporalThreshold = 0.6;
params.waveCount = 500;
params.backSp = 51;
params.forwardSp = 51;

resorter(params)
quit;