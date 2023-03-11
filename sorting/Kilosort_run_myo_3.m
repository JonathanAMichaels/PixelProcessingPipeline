script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))

try
    restoredefaultpath
end
dbstop if error

chanMapFile = myo_chan_map_file
disp(['Using this channel map: ' chanMapFile])

addpath(genpath([script_dir '/sorting/Kilosort-3.0']))
addpath(genpath([script_dir '/sorting/npy-matlab']))

run([script_dir '/sorting/Kilosort_config_3.m']);
ops.fbinary = fullfile(myomatrix_folder, 'data.bin');
ops.fproc = fullfile(myomatrix_folder, 'proc.dat');
ops.brokenChan = fullfile(myomatrix_folder, 'brokenChan.mat');
ops.chanMap = fullfile(chanMapFile);
ops.NchanTOT = double(num_chans);

ops.nt0 = 201;
ops.NT = 2 * 64 * 1024 + ops.ntbuff;
ops.sigmaMask = Inf; % we don't want a distance-dependant decay
ops.Th = [9 8];
ops.nfilt_factor = 4;
ops.nblocks = 0;
ops.nt0min = ceil(ops.nt0 / 2);
ops.nPCs = 6;
ops.nEig = 3;

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
rezToPhy2(rez, myomatrix_folder);
save(fullfile(script_dir, '/tmp/ops.mat'), 'ops')

% delete(ops.fproc);

quit;
