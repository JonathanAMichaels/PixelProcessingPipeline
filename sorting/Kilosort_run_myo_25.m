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

addpath(genpath([script_dir '/sorting/Kilosort-2.5']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

run([script_dir '/sorting/Kilosort_config_25.m']);
ops.fbinary = fullfile(myomatrix_folder, 'data.bin');
ops.fproc   = fullfile(myomatrix_folder, 'proc.dat');
ops.brokenChan = fullfile(myomatrix_folder, 'brokenChan.mat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = double(num_chans);

ops.nt0 = 61;
%ops.NT = 64*1024 + ops.ntbuff;
%ops.nskip           = 10;  % how many batches to skip for determining spike PCs
%ops.nSkipCov            = 10; % compute whitening matrix from every N-th batch
ops.sigmaMask = Inf; % we don't want a distance-dependant decay
ops.Th = [10 4];
ops.nfilt_factor = 4;
ops.nblocks = 0;
ops.nt0min = ceil(ops.nt0/2);
ops.nPCs = 6;
ops.nEig = 3;

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

ops

rez                = preprocessDataSub(ops);

rez                = datashift2(rez, 1);


% ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
iseed = 1;

% main tracking and template matching algorithm
rez = learnAndSolve8b(rez, iseed);

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort/issues/29
%rez = remove_ks2_duplicate_spikes(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% decide on cutoff
rez = set_cutoff(rez);
% eliminate widely spread waveforms (likely noise)
rez.good = get_good_units(rez);

fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy(rez, myomatrix_folder);

%% if you want to save the results to a Matlab file...

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% final time sorting of spikes, for apps that use st3 directly
[~, isort]   = sortrows(rez.st3);
rez.st3      = rez.st3(isort, :);

% Ensure all GPU arrays are transferred to CPU side before saving to .mat
rez_fields = fieldnames(rez);
for i = 1:numel(rez_fields)
    field_name = rez_fields{i};
    if(isa(rez.(field_name), 'gpuArray'))
        rez.(field_name) = gather(rez.(field_name));
    end
end

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(myomatrix_folder, 'rez2.mat');
save(fname, 'rez', '-v7.3');

quit;