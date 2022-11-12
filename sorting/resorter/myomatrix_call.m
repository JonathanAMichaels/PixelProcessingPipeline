load('/tmp/config.mat')
if num_chans == 16
    load([script_dir '/geometries/bipolar_test_kilosortChanMap'])
elseif num_chans == 32
    load([script_dir '/geometries/monopolar_test_kilosortChanMap'])
end
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = myomatrix_folder;
params.binaryFile = [myomatrix_folder '/proc.dat'];
params.userSorted = false;
params.savePlots = true;
params.waveCount = 800;
params.consistencyThreshold = 0;
params.multiSNRThreshold = 5;

resorter(params)
quit;