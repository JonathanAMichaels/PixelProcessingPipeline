load('/tmp/config.mat')

% for use with monopolar
channelRemap = [23:-1:8 24:31 0:7] + 1;
% for use with bipolar
channelLabelsBipolar = [25, 26; 27, 28; 29, 30; 31, 32; ...
    16, 15; 14, 13; 12, 11; 10, 9; 8, 7; 6, 5; 4, 3; 2, 1; ...
    17, 18; 19, 20; 21, 22; 23, 24];

pathToYourConfigFile = [script_dir '/sorting/Kilosort_config.m'];

if type == 1
    phyDir = 'phyData';
    chanMapFile = [script_dir '/geometries/neuropixPhase3B1_kilosortChanMap.mat'];
else
    chanList = Session.myo_chan_list(myomatrix_number,1) : Session.myo_chan_list(myomatrix_number,2);
    if length(chanList) == 16
        chanMapFile = [script_dir '/geometries/bipolar_test_kilosortChanMap.mat'];
    elseif length(chanList) == 32
        chanMapFile = [script_dir '/geometries/monopolar_test_kilosortChanMap.mat'];
    else
        error('Channel map not implemented')
    end    
    if myomatrix_number == 1
        phyDir = 'phyDataMyo';
    else
        phyDir = ['phyDataMyo-' num2str(myomatrix_number)];
    end
end

disp(['Using this channel map: ' chanMapFile])

rootZ = [neuropixel_folder '/'];
rootH = [rootZ phyDir '/'];
mkdir(rootH);

if Session.trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = Session.trange;
end

ops.NchanTOT  = 385; % total number of channels in your recording
run(pathToYourConfigFile);
ops.fproc   = fullfile(rootH, 'temp_wh.dat');
ops.chanMap = fullfile(chanMapFile);

if type == 2
    if myomatrix_number == 1
        checkFile = [rootZ 'MyomatrixData.bin'];
    else
        checkFile = [rootZ 'MyomatrixData-' num2str(myomatrix_number) '.bin'];
    end    
    if ~isfile(checkFile)
        disp('Did not find myomatrix binary, building it')
        prefix = Session.myo_prefix;
        dataChan = chanList;
   
        tempdata = load_open_ephys_data([rootZ '100_' prefix num2str(dataChan(1)) '.continuous']);
        tL = length(tempdata);
        clear tempdata
        data = zeros(tL, length(dataChan), 'int16');
        for chan = 1:length(dataChan)
            data(:,chan) = load_open_ephys_data([rootZ '100_' prefix num2str(dataChan(chan)) '.continuous']);
        end
        if length(dataChan) == 32
            data = data(:,channelRemap);
        end
        analogData = load_open_ephys_data([rootZ '100_' num2str(Session.myo_analog_chan) '.continuous']);
        analogData(analogData > 5) = 5;
        sync = logical(round(analogData / max(analogData)));
        clear analogData
        
        clf
        for q = 1:2
            if q == 1
                [b, a] = butter(2, [350 7500] / (30000/2), 'bandpass');
            elseif q == 2
                [b, a] = butter(2, [10 200] / (30000/2), 'bandpass');
            end
            tRange = size(data,1) - (30000*60*3) : size(data,1) - (30000*60);
            data_filt = zeros(length(tRange),size(data,2));
            for i = 1:size(data,2)
                data_filt(:,i) = filtfilt(b, a, double(data(tRange,i)));
            end
            subplot(1,2,q)
            hold on
            for i = 1:size(data,2)
                plot(data_filt(:,i) + i*2000)
            end
        end
        S = std(data_filt,[],1);
        if length(dataChan) == 32
            brokenChan = find(S > 150);
        elseif length(dataChan) == 16
            brokenChan = find(S > 20);
        end
        disp('Broken channels are:')
        brokenChan
        data(:,brokenChan) = 0;
        clear data_filt
        
        % Generate "Bulk EMG" dataset
        notBroken = 1:size(data,2);
        notBroken(brokenChan) = [];
        if length(dataChan) == 32
            bottomHalf = [9:16 25:32];
            topHalf = [1:8 17:24];
            bottomHalf(ismember(bottomHalf, brokenChan)) = [];
            topHalf(ismember(topHalf, brokenChan)) = [];
            bEMG = int16(mean(data(:,bottomHalf),2)) - int16(mean(data(:,topHalf),2));
        else
            bEMG = int16(mean(data(:,notBroken),2));
        end
        
        if myomatrix_number == 1
            fileID = fopen([rootZ 'MyomatrixData.bin'], 'w');
            save([rootZ 'bulkEMG'], 'bEMG', 'notBroken', 'dataChan')
        else
            fileID = fopen([rootZ 'MyomatrixData' num2str(myomatrix_number) '.bin'], 'w');
            save([rootZ 'bulkEMG-' num2str(myomatrix_number)], 'bEMG', 'notBroken', 'dataChan')
        end
        clear bEMG
        disp('Saved generated bulk EMG')
        fwrite(fileID, data', 'int16');
        fclose(fileID);
        clear data
        disp('Saved myomatrix data binary')
        save([rootZ 'sync'], 'sync')
        clear sync
        disp('Saved sync data')
    end
end

% find the binary file
if type == 1
    fs          = dir(fullfile([rootZ 'NeuropixelsRegistration/registered/'], '*.bin'));
    ops.fbinary = fullfile([rootZ 'NeuropixelsRegistration/registered/'], fs(1).name);
    overlap_s = Sorting.Neuro_sorting.overlap_s;
    channel_sep = Sorting.Neuro_sorting.channel_sep;
    ops.CAR = 1;
else
    fs          = dir(fullfile(rootZ, 'Myo*.bin'));
    ops.fbinary = fullfile(rootZ, fs(myomatrix_number).name);
    ops.NchanTOT  = length(chanList); % total number of channels in your recording
    if ops.NchanTOT == 16
        ops.CAR = 0;
    else
        ops.CAR = 1;
    end
    overlap_s = Sorting.Myo_sorting.overlap_s;
    channel_sep = Sorting.Myo_sorting.channel_sep;
    ops.sigmaMask = 1200;
end
disp(['Using ' ops.fbinary])

rez                = preprocessDataSub(ops);
disp('Finished preprocessing')
rez                = datashift2(rez, 1);
disp('Finished datashift')
[rez, st3, tF]     = extract_spikes(rez);
disp('Finished extract spikes')
rez                = template_learning(rez, tF, st3);
disp('Finished template learning')
clear st3 tF
[rez, st3, tF]     = trackAndSort(rez);
disp('Finished track and sort')
rez                = final_clustering(rez, tF, st3);
disp('Finished final clustering')

rez = remove_ks2_duplicate_spikes(rez, 'overlap_s', overlap_s, 'channel_separation_um', channel_sep);
disp('Finished removing duplicates')
rez                = find_merges(rez, 1);
disp('Finished merges')

fprintf('found %d good units \n', sum(rez.good>0))

rootQ = fullfile(rootZ, 'kilosort3');
if myomatrix_number ~= 1
    rootQ = fullfile(rootZ, ['kilosort3-' num2str(myomatrix_number)]);
end
mkdir(rootQ)
cd(rootQ)
fprintf('Saving results to Phy  \n')
rezToPhy2(rez, pwd);
quit;