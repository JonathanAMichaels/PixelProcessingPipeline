function rez = Kilosort_run_czuba_myo(ops_input_params)
    dbstop if error
    script_dir = pwd; % get directory where repo exists
    load(fullfile(script_dir, '/tmp/config.mat'))
    load(fullfile(myo_sorted_dir, 'brokenChan.mat'))

    % set GPU to use
    disp(strcat("Setting GPU device to use: ", num2str(GPU_to_use)))
    gpuDevice(GPU_to_use);

    % get and set channel map
    if ~isempty(brokenChan) && remove_bad_myo_chans(1) ~= false
        chanMapFile = fullfile(myo_sorted_dir, 'chanMapAdjusted.mat');
    else
        chanMapFile = myo_chan_map_file;
    end
    disp(['Using this channel map: ' chanMapFile])

    % set paths
    try
        restoredefaultpath
    end
    addpath(genpath([script_dir '/sorting/Kilosort-3.0']))
    addpath(genpath([script_dir '/sorting/npy-matlab']))

    % phyDir = 'sorted-czuba';
    % rootZ = [neuropixel_folder '/'];
    % rootH = [rootZ phyDir '/'];
    % mkdir(rootH);

    run([script_dir '/sorting/Kilosort_config_czuba.m']);
    % ops.fbinary = fullfile(neuropixel);
    % ops.fproc = fullfile(rootH, 'proc.dat');
    % ops.chanMap = fullfile(chanMapFile);
    % ops.NchanTOT = 385;
    % ops.saveDir = rootH;
    ops.saveDir = myo_sorted_dir; % set directory for writes
    ops.fbinary = fullfile(ops.saveDir, 'data.bin');
    ops.fproc = fullfile(ops.saveDir, 'proc.dat');
    ops.brokenChan = fullfile(ops.saveDir, 'brokenChan.mat');
    ops.chanMap = fullfile(chanMapFile);
    ops.nt0 = 61;
    ops.ntbuff = 1024; %ceil(bufferSec * ops.fs / 64) * 64; %  ceil(batchSec/4*ops.fs/64)*64; % (def=64)
    ops.NT = 2048 * 32 + ops.ntbuff; %ceil(batchSec * ops.fs / 32) * 32; % convert to 32 count increments of samples
    ops.sigmaMask = Inf; % we don't want a distance-dependant decay
    ops.nPCs = double(num_KS_components); % how many PCs to project the spikes into (also used as number of template prototypes)
    ops.nTEMP = ops.nPCs; % number of templates to use for spike detection
    ops.nEig = ops.nPCs; % rank of svd for templates, % keep same as nPCs to avoid error
    ops.NchanTOT = double(max(num_chans - length(brokenChan), ops.nEig));
    ops.Th = [6 2]; % threshold crossings for pre-clustering (in PCA projection space)
    ops.spkTh = -6; % spike threshold in standard deviations (-6 default) (used in isolated_peaks_new/buffered and spikedetector3PC.cu)
    ops.nfilt_factor = ops.nPCs; % max number of clusters per good channel in a batch (even temporary ones)
    ops.nblocks = 0;
    ops.nt0min = ceil(ops.nt0 / 2); % peak of template match will be this many points away from beginning
    ops.nskip = 1; % how many batches to skip for determining spike PCs
    ops.nSkipCov = 1; % compute whitening matrix and prototype templates using every N-th batch
    ops.lam = 10; % amplitude penalty (0 means not used, 10 is average, 50 is a lot)
    ops.CAR = 0; % whether to perform CAR
    ops.loc_range = [5 1]; % [timepoints channels], area to detect peaks; plus/minus for both time and channel. Doing abs() of data during peak isolation, so using 4 instead of default 5. Only 1 channel to avoid elimination of waves
    ops.long_range = [ops.nt0min 1]; % [timepoints channels], range within to use only the largest peak
    ops.fig = 1; % whether to plot figures
    ops.recordings = recordings;
    ops.momentum = [20 400];
    ops.clipMin = 200; % clip template updating to a minimum number of contributing spikes
    ops.clipMinFit = .8;
    %batchSec = 10; % number of seconds in each batch     (TBC: 8:10 seems good for 1-2 hr files and/or 32 channels)
    %bufferSec = 2; % define number of seconds of data for buffer
    % sample from batches more sparsely (in certain circumstances/analyses)
    % batchSkips = ceil(60 / batchSec); % do high-level assessments at least once every minute of data

    %% gridsearch section
    % only try to use gridsearch values if ops_input_params is a struct and fields are present
    if isa(ops_input_params, 'struct') && ~isempty(fieldnames(ops_input_params))
        % Combine input ops into the existing ops struct
        fields = fieldnames(ops_input_params);
        for iField = 1:size(fields, 1)
            ops.(fields{iField}) = ops_input_params.(fields{iField});
        end
        % ops.NT = ops.nt0 * 32 + ops.ntbuff; % 2*87040 % 1024*(32+ops.ntbuff);
    end
    %% end gridsearch section

    disp(['Using ' ops.fbinary])

    if trange(2) == 0
        ops.trange = [0 Inf];
    else
        ops.trange = trange;
    end

    % disp(ops)
    % rez = preprocessDataSub(ops);

    % rez = datashift2(rez, 1);

    % rez.W = []; rez.U = [];, rez.mu = [];
    % rez = learnAndSolve8b(rez, now);

    % % OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
    % % See issue 29: https://github.com/MouseLand/Kilosort2/issues/29
    % %rez = remove_ks2_duplicate_spikes(rez);

    % % final merges
    % rez = find_merges(rez, 1);

    % % final splits by SVD
    % %rez = splitAllClusters(rez, 1);

    % % final splits by amplitudes
    % rez = splitAllClusters(rez, 0);

    % % decide on cutoff
    % rez = set_cutoff(rez, 1);

    % [rez.good, ~] = get_good_units(rez);

    % fprintf('found %d good units \n', sum(rez.good > 0))

    rez = preprocessDataSub(ops);
    ops.channelDelays = rez.ops.channelDelays;
    rez = datashift2(rez, 1);
    [rez, st3, tF] = extract_spikes(rez);
    %%% plots
    figure(5);
    plot(st3(:, 1), '.')
    title('Spike times versus spike ID')
    figure(6);
    plot(st3(:, 2), '.')
    title('Upsampled grid location of best template match spike ID')
    figure(7);
    plot(st3(:, 3), '.')
    title('Amplitude of template match for each spike ID')
    figure(8); hold on;
    plot(st3(:, 4), 'g.')
    for kSpatialDecay = 1:6
        less_than_idx = find(st3(:, 4) < 6 * kSpatialDecay);
        more_than_idx = find(st3(:, 4) >= 6 * (kSpatialDecay - 1));
        idx = intersect(less_than_idx, more_than_idx);
        bit_idx = bitand(st3(:, 4) < 6 * kSpatialDecay, st3(:, 4) >= 6 * (kSpatialDecay - 1));
        plot(idx, st3(bit_idx, 4), '.')
    end
    title('Prototype templates for each spatial decay value (1:6:30) resulting in each best match spike ID')
    figure(9);
    plot(st3(:, 5), '.')
    title('Amplitude of template match for each spike ID (Duplicate of st3(:,3))')
    figure(10);
    plot(st3(:, 6), '.')
    title('Batch ID versus spike ID')
    figure(11);
    for iTemp = 1:size(tF, 2)
        subplot(size(tF, 2), 1, iTemp)
        plot(squeeze(tF(:, iTemp, :)), '.')
    end
    %%% end plots
    [rez, ~] = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);
    % plot_templates_on_raw_data_fast(rez, st3);
    rez = final_clustering(rez, tF, st3);
    rez = find_merges(rez, 1);

    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy2(rez, ops.saveDir);
    save(fullfile(ops.saveDir, '/ops.mat'), '-struct', 'ops');
    
    ops

    % quit;
end
