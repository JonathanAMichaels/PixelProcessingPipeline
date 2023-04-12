script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))
load(fullfile(myo_sorted_dir, 'brokenChan.mat'))
chanMapFile = myo_chan_map_file
disp(['Using this channel map: ' chanMapFile])

% load and modify channel map variables to remove broken channel elements, if desired
if length(brokenChan) > 0 && remove_bad_myo_chans(1) ~= false
    load(chanMapFile)
    disp('Broken channels were just removed from that channel map')
    load(myo_chan_map_file)
    chanMap(end-length(brokenChan)+1:end) = []; % take off end to save indexing
    chanMap0ind(end-length(brokenChan)+1:end) = []; % take off end to save indexing
    connected(brokenChan) = [];
    kcoords(brokenChan) = [];
    xcoords(brokenChan) = [];
    ycoords(brokenChan) = [];
    save(fullfile(myo_sorted_dir, 'chanMap_minus_brokenChans.mat'), 'chanMap', 'connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs', 'name')
    chanMapFile = fullfile(myo_sorted_dir, 'chanMap_minus_brokenChans.mat');
end

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

% delete(ops.fproc);

quit;
