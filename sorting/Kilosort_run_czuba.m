script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))

dbstop if error

try
    restoredefaultpath
end

addpath(genpath([script_dir '/sorting/Kilosort-czuba']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

phyDir = 'sorted-czuba';
rootZ = [neuropixel_folder '/'];
rootH = [rootZ phyDir '/'];
mkdir(rootH);

chanMapFile = [script_dir '/geometries/neuropixPhase3B1_kilosortChanMap.mat'];
disp(['Using this channel map: ' chanMapFile])

run([script_dir '/sorting/Kilosort_config_czuba.m']);
ops.fbinary = fullfile(neuropixel);
ops.fproc = fullfile(rootH, 'proc.dat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = 385;
ops.saveDir = rootH;

disp(['Using ' ops.fbinary])

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

ops.chanMap
rez = preprocessDataSub(ops);

rez = datashift2(rez, 1);

rez.W = []; rez.U = [];, rez.mu = [];
rez = learnAndSolve8b(rez, now);

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort2/issues/29
%rez = remove_ks2_duplicate_spikes(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
%rez = splitAllClusters(rez, 1);

% final splits by amplitudes
rez = splitAllClusters(rez, 0);

% decide on cutoff
rez = set_cutoff(rez);

[rez.good, ~] = get_good_units(rez);

fprintf('found %d good units \n', sum(rez.good > 0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, rootH);

%delete(ops.fproc)

quit;
