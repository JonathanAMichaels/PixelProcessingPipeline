load('/tmp/config.mat')
% if num_chans == 16
%     % load([script_dir '/geometries/bipolar_test_kilosortChanMap'])
%     load([script_dir '/geometries/linear_16ch_RF400_kilosortChanMap'])
% elseif num_chans == 32
%     load([script_dir '/geometries/monopolar_test_kilosortChanMap'])
% end
load(myo_chan_map_file)
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = [myomatrix_folder '/custom_merge'];
params.binaryFile = [myomatrix_folder '/data.bin'];
params.savePlots = true;
params.waveCount = 1000;
params.skipFilter = false;
% make sure a sorting exists
if isfile([myomatrix_folder '/spike_times.npy'])
    resorter(params)
else
    disp('No spike sorting to post-process')
end

quit;