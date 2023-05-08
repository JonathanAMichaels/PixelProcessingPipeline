function Kilosort_run_myo_3(ops_input_params)
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
    ops.nt0 = 201;
    ops.ntbuff = ops.nt0 + 3;
    ops.NT = 2048 * 32 + ops.ntbuff;
    ops.sigmaMask = Inf; % we don't want a distance-dependant decay
    ops.Th = [9 8];
    ops.nfilt_factor = 4;
    ops.nblocks = 0;
    ops.nt0min = ceil(ops.nt0 / 2);
    ops.nPCs = 6;
    ops.nEig = 3;
    ops.lam = 10; % amplitude penalty (0 means not used, 10 is average, 50 is a lot)
    ops.ThPre = 8; % threshold crossings for pre-clustering (in PCA projection space)
    ops.CAR = 0; % whether to perform CAR
    ops.loc_range = [5 4]; % area to detect peaks; plus/minus for both time and channel
    ops.long_range = [30 6]; % range to detect isolated peaks: [timepoints channels]
    ops.fig = 0; % whether to plot figures
    ops.nSkipCov = 25; % compute whitening matrix and prototype templates using every N-th batch

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
    rez = template_learning(rez, tF, st3);
    [rez, st3, tF] = trackAndSort(rez);
    rez = final_clustering(rez, tF, st3);
    rez = find_merges(rez, 1);

    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy2(rez, myo_sorted_dir);
    save(fullfile(myo_sorted_dir, '/ops.mat'), 'ops')
    % create timestamped backup folder for this run
    % split_sorted_folder_name = split(myo_sorted_dir, filesep);
    % sorted_folder_suffix = split_sorted_folder_name{end};
    % copyfile(myo_sorted_dir,
    % fullfile(myo_sorted_dir, '..', [sorted_folder_suffix '_' datestr(now, 'yyyy-mm-dd_HH:MM:SS')]))

    quit;
end
