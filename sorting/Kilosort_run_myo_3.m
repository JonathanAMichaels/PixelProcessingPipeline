function rez = Kilosort_run_myo_3(ops_input_params)
    dbstop if error
    script_dir = pwd; % get directory where repo exists
    load(fullfile(script_dir, '/tmp/config.mat'))
    load(fullfile(myo_sorted_dir, 'brokenChan.mat'))

    if length(brokenChan) > 0 && remove_bad_myo_chans(1) ~= false
        chanMapFile = fullfile(myo_sorted_dir, 'chanMap_minus_brokenChans.mat');
    else
        chanMapFile = myo_chan_map_file;
    end
    disp(['Using this channel map: ' chanMapFile])

    try
        restoredefaultpath
    end

    addpath(genpath([script_dir '/sorting/Kilosort-3.0']))
    addpath(genpath([script_dir '/sorting/npy-matlab']))

    run([script_dir '/sorting/Kilosort_config_3.m']);

    ops.fbinary = fullfile(myo_sorted_dir, 'data.bin');
    ops.fproc = fullfile(myo_sorted_dir, 'proc.dat');
    ops.brokenChan = fullfile(myo_sorted_dir, 'brokenChan.mat');
    ops.chanMap = fullfile(chanMapFile);
    ops.NchanTOT = double(num_chans - length(brokenChan));
    ops.nt0 = 61;
    ops.ntbuff = 512; % defined as 64;
    ops.NT = 2048 * 32 + ops.ntbuff; % convert to 32 count increments of samples % defined as 2048 * 32 + ops.ntbuff;
    ops.sigmaMask = Inf; % we don't want a distance-dependant decay
    ops.Th = [9 8]; % threshold crossings for pre-clustering (in PCA projection space)
    ops.spkTh = -2; % spike threshold in standard deviations (-6 default) (only used in isolated_peaks_new)
    ops.nfilt_factor = 12; % max number of clusters per good channel (even temporary ones)
    ops.nblocks = 0;
    ops.nt0min = ceil(ops.nt0 / 2); % peak of template match will be this many points away from beginning
    ops.nPCs = 6; % how many PCs to project the spikes into (also used as number of template prototypes)
    ops.nskip = 1; % how many batches to skip for determining spike PCs
    ops.nSkipCov = 1; % compute whitening matrix and prototype templates using every N-th batch
    ops.nEig = 3;
    ops.lam = 15; % amplitude penalty (0 means not used, 10 is average, 50 is a lot)
    ops.CAR = 0; % whether to perform CAR
    ops.loc_range = [5 1]; % area to detect peaks; plus/minus for both time and channel
    ops.long_range = [ops.nt0min 1]; % range to detect isolated peaks: [timepoints channels]
    ops.fig = 1; % whether to plot figures
    ops.recordings = recordings;

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

    if trange(2) == 0
        ops.trange = [0 Inf];
    else
        ops.trange = trange;
    end

    ops

    rez = preprocessDataSub(ops);
    rez = datashift2(rez, 1);
    [rez, st3, tF] = extract_spikes(rez);
    %%% plots
    % figure(5);
    % plot(st3(:, 1), '.')
    % title('Spike times versus spike ID')
    % figure(6);
    % plot(st3(:, 2), '.')
    % title('Upsampled grid location of best template match spike ID')
    % figure(7);
    % plot(st3(:, 3), '.')
    % title('Amplitude of template match for each spike ID')
    % figure(8); hold on;
    % plot(st3(:, 4), 'g.')
    % for kSpatialDecay = 1:6
    %     less_than_idx = find(st3(:, 4) < 6 * kSpatialDecay);
    %     more_than_idx = find(st3(:, 4) >= 6 * (kSpatialDecay - 1));
    %     idx = intersect(less_than_idx, more_than_idx);
    %     bit_idx = bitand(st3(:, 4) < 6 * kSpatialDecay, st3(:, 4) >= 6 * (kSpatialDecay - 1));
    %     plot(idx, st3(bit_idx, 4), '.')
    % end
    % title('Prototype templates for each spatial decay value (1:6:30) resulting in each best match spike ID')
    % figure(9);
    % plot(st3(:, 5), '.')
    % title('Amplitude of template match for each spike ID (Duplicate of st3(:,3))')
    % figure(10);
    % plot(st3(:, 6), '.')
    % title('Batch ID versus spike ID')
    % figure(11);
    % for iTemp = 1:size(tF, 2)
    %     subplot(size(tF, 2), 1, iTemp)
    %     plot(squeeze(tF(:, iTemp, :)), '.')
    % end
    %%% end plots
    [rez, ~]  = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);
    % plot_templates_on_raw_data_fast(rez, st3);
    rez = final_clustering(rez, tF, st3);
    rez = find_merges(rez, 1);
    
    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy2(rez, myo_sorted_dir);
    save(fullfile(myo_sorted_dir, '/ops.mat'), 'ops')

    % quit;
end
