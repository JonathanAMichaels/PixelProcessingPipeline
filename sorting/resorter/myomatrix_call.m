script_dir = pwd
load(fullfile(script_dir, '/tmp/config.mat'))
load(myo_chan_map_file)
params.chanMap = cat(2, xcoords, ycoords);
params.kiloDir = [myomatrix_folder '/custom_merge'];
params.binaryFile = [myomatrix_folder '/data.bin'];
params.savePlots = true;
params.skipFilter = false;
% make sure a sorting exists
if isfile([myomatrix_folder '/spike_times.npy'])
    resorter(params)
else
    disp('No spike sorting to post-process')
end

quit;