script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))

try
    restoredefaultpath
end

addpath(genpath([script_dir '/sorting/Kilosort-2.0']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

chanMapFile = [script_dir '/geometries/neuropixPhase3B1_kilosortChanMap.mat'];
disp(['Using this channel map: ' chanMapFile])

phyDir = 'sorted';
rootZ = [neuropixel_folder '/'];
rootH = [rootZ phyDir '/'];
mkdir(rootH);

run([script_dir '/sorting/Kilosort_config_2.m']);
ops.fbinary = fullfile(rootH, 'proc.dat');
ops.fproc = fullfile(rootH, 'proc2.dat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = 384; % 385

disp(['Using ' ops.fbinary])

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end


% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

% time-reordering as a function of drift
rez = clusterSingleBatches(rez);

% main tracking and template matching algorithm
rez = learnAndSolve8b(rez);

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort2/issues/29
rez = remove_ks2_duplicate_spikes(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% final splits by amplitudes
rez = splitAllClusters(rez, 0);

% decide on cutoff
rez = set_cutoff(rez);

fprintf('found %d good units \n', sum(rez.good > 0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, rootH);

delete(ops.fproc)

quit;
