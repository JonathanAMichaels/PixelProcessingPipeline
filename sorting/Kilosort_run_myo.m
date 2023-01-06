load('/tmp/config.mat')

try
    restoredefaultpath
end
dbstop if error

if num_chans == 16
    chanMapFile = [script_dir '/geometries/bipolar_test_kilosortChanMap.mat'];
elseif num_chans == 32
    chanMapFile = [script_dir '/geometries/monopolar_test_kilosortChanMap.mat'];
end
disp(['Using this channel map: ' chanMapFile])

addpath(genpath([script_dir '/sorting/Kilosort-2.0']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

run([script_dir '/sorting/Kilosort_config_2.m']);
ops.fbinary = fullfile(myomatrix_folder, 'data.bin');
ops.fproc   = fullfile(myomatrix_folder, 'proc.dat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = double(num_chans);

ops.nt0 = 61;
%ops.Th = [2 2]
%ops.spkTh = -3;
ops.minFR = 0.01;
ops.NT = 4*512*1024+ ops.ntbuff;
ops.nskip           = 2;  % how many batches to skip for determining spike PCs
ops.nSkipCov            = 2; % compute whitening matrix from every N-th batch
ops.reorder = 1;

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

ops

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

fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, myomatrix_folder);

delete(ops.fproc)

quit;