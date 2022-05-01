load('/tmp/config.mat')

% for use with monopolar
channelRemap = [23:-1:8 24:31 0:7] + 1;
% for use with bipolar
channelLabelsBipolar = [25, 26; 27, 28; 29, 30; 31, 32; ...
    16, 15; 14, 13; 12, 11; 10, 9; 8, 7; 6, 5; 4, 3; 2, 1; ...
    17, 18; 19, 20; 21, 22; 23, 24];

pathToYourConfigFile = [script_dir '/sorting/Kilosort_config.m'];

if type == 1
    file = neuropixel;
    phyDir = 'phyData';
    chanMapFile = [script_dir '/geometries/neuropixPhase3B1_kilosortChanMap.mat'];
else
    file = myomatrix;
end
chanList = channel_list;


if type == 2
    if length(innerChanList) == 16
        chanMapFile = [script_dir '/geormetries/bipolar_test_kilosortChanMap.mat'];
    elseif length(innerChanList) == 32
        chanMapFile = [script_dir '/geormetries/monopolar_test_kilosortChanMap.mat'];
    else
        error('What.')
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
mkdir(rootH)

if Session.trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = Session.trange;
end

ops.NchanTOT  = 385; % total number of channels in your recording
run(pathToYourConfigFile)
ops.fproc   = fullfile(rootH, 'temp_wh.dat');
ops.chanMap = fullfile(chanMapFile);

if type == 2
    checkFile = [rootZ 'MyomatrixData.bin'];
    if ~isfile(checkFile)
        prefix = prefixList{subject};
        postfix = postfixList{subject};
        dataChan = innerChanList;
        
        tempdata = cell(1,length(dataChan));
        for chan = 1:length(dataChan)
            tempdata{chan} = load_open_ephys_data([rootZ '100_' prefix num2str(dataChan(chan)) postfix '.continuous']);
        end
        data = zeros(size(tempdata{1},1), length(dataChan));
        for chan = 1:size(data,2)
            data(:,chan) = tempdata{chan};
        end
        clear tempdata
        if length(dataChan) == 32
            data = data(:,channelRemap);
        end
        analogData = load_open_ephys_data([rootZ '100_' analogChanList{subject} postfix '.continuous']);
        analogData(analogData > 5) = 5;
        sync = logical(round(analogData / max(analogData)));
        
        [b, a] = butter(4, [350 7500] / (30000/2), 'bandpass');
        tRange = size(data,1) - 2000000 : size(data,1);
        data_filt = zeros(length(tRange),size(data,2));
        for i = 1:size(data,2)
            data_filt(:,i) = filtfilt(b, a, data(tRange,i));
        end
        hold on
        for i = 1:size(data,2)
            plot(data_filt(:,i) + i*500)
        end
        drawnow
        pause
        
        data(:,innerBrokenList) = 0;
        dataSave = int16(data');
        whos dataSave
        if myomatrix_number == 1
            fileID = fopen([rootZ 'MyomatrixData.bin'], 'w');
        else
            fileID = fopen([rootZ 'MyomatrixData' num2str(myomatrix_number) '.bin'], 'w');
        end
        fwrite(fileID, dataSave, 'int16');
        fclose(fileID);
        save([rootZ 'syncData'], 'sync')
        clear data dataSave
    end
end

% find the binary file
if type == 1
    fs          = dir(fullfile([rootZ 'NeuropixelsRegistration/registered/'], '*.bin'));
    ops.fbinary = fullfile([rootZ 'NeuropixelsRegistration/registered/'], fs(1).name);
    channelSep = 100; % Default for Neuropixels
    overlap_s = Sorting.Neuro_sorting.overlap_s;
    ops.CAR = 1;
else
    fs          = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*.dat'))];
    ops.fbinary = fullfile(rootZ, fs(rr).name);
    ops.NchanTOT  = length(innerChanList); % total number of channels in your recording
    if ops.NchanTOT == 16
        ops.CAR = 0;
    else
        ops.CAR = 1;
    end
    overlap_s = Sorting.Myo_sorting.overlap_s;
    ops.sigmaMask = 1200;
end
disp(ops.fbinary)

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

%ind = find(rez.st3(:,2) == 0);
%rez.st3(ind,:) = [];
%rez.xy(ind,:) = [];
%disp([num2str(length(ind)) ' deleted'])

rez = remove_ks2_duplicate_spikes(rez, 'overlap_s', overlap_s, 'channel_separation_um', channelSep);
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
