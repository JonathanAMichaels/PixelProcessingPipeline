script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))
load(fullfile(myo_sorted_dir, 'brokenChan.mat'))

% load and modify channel map variables to remove broken channel elements, if desired
if length(brokenChan) > 0 && remove_bad_myo_chans(1) ~= false
    chanMapFile = fullfile(myo_sorted_dir, 'chanMap_minus_brokenChans.mat');
else
    chanMapFile = myo_chan_map_file;
end
disp(['Using this channel map: ' chanMapFile])

try
    restoredefaultpath
end
dbstop if error

addpath(genpath([script_dir '/sorting/Kilosort-3.0']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

run([script_dir '/sorting/Kilosort_config_3.m']);
ops.fbinary = fullfile(myo_sorted_dir, 'data.bin');
ops.fproc = fullfile(myo_sorted_dir, 'proc.dat');
ops.brokenChan = fullfile(myo_sorted_dir, 'brokenChan.mat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = double(num_chans - length(brokenChan));
ops.nt0 = 201;
ops.NT = 2 * 64 * 1024 + ops.ntbuff;
ops.sigmaMask = Inf; % we don't want a distance-dependant decay
ops.Th = [9 8];
ops.nfilt_factor = 4;
ops.nblocks = 0;
ops.nt0min = ceil(ops.nt0 / 2);
ops.nPCs = 6;
ops.nEig = 3;
ops.lam = 10; % amplitude penalty (0 means not used, 10 is average, 50 is a lot)
ops.ThPre = 8; % threshold crossings for pre-clustering (in PCA projection space)

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

ops

rez = preprocessDataSub(ops);
rez = datashift2(rez, 1);
[rez, st3, tF] = extract_spikes(rez);
rez = template_learning(rez, tF, st3);
[rez, st3, tF] = trackAndSort(rez);
rez = final_clustering(rez, tF, st3);
rez = find_merges(rez, 1);

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy2(rez, myo_sorted_dir);
save(fullfile(myo_sorted_dir, '/ops.mat'), 'ops')

%%% create timestamped backup folder for each run, so that these results in sorted0 don't get overwritten later
% split_sorted_folder_name = split(myo_sorted_dir, filesep);
% sorted_folder_suffix = split_sorted_folder_name{end};
% copyfile(myo_sorted_dir, fullfile(myo_sorted_dir, '..', [sorted_folder_suffix '_' datestr(now, 'yyyy-mm-dd_HH:MM:SS')]))

quit;
