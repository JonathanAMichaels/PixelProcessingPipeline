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

addpath(genpath([script_dir '/sorting/Kilosort-3.0']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

run([script_dir '/sorting/Kilosort_config_3.m']);
ops.fbinary = fullfile(myomatrix_folder, 'data.bin');
ops.fproc   = fullfile(myomatrix_folder, 'proc.dat');
ops.brokenChan = fullfile(myomatrix_folder, 'brokenChan.mat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = double(num_chans);

ops.nt0 = 151;
ops.NT = 64*1024 + ops.ntbuff;
ops.nskip           = 10;  % how many batches to skip for determining spike PCs
ops.nSkipCov            = 10; % compute whitening matrix from every N-th batch
ops.sigmaMask = Inf; % we don't want a distance-dependant decay
ops.Th = [9 8];
ops.nfilt_factor = 4;
ops.nblocks = 0;
ops.nt0min = ceil(ops.nt0/2);
ops.nPCs = 12;
ops.nEig = 6;

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

ops

rez                = preprocessDataSub(ops);

rez                = datashift2(rez, 1);

[rez, st3, tF]     = extract_spikes(rez);

rez                = template_learning(rez, tF, st3);
rez.U(:,1,end)

[rez, st3, tF]     = trackAndSort(rez);
rez.U(:,1,end)
rez                = final_clustering(rez, tF, st3);
rez.U(:,1,end)
rez                = find_merges(rez, 1);
rez.U(:,1,end)
% write to Phy
fprintf('Saving results to Phy  \n')
rezToPhy2(rez, myomatrix_folder);

%delete(ops.fproc);

quit;