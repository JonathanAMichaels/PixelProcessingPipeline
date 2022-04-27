close all
clear

% for use with monopolar
channelRemap = [23:-1:8 24:31 0:7] + 1;
% for use with bipolar
channelLabelsBipolar = [25, 26; 27, 28; 29, 30; 31, 32; ...
    16, 15; 14, 13; 12, 11; 10, 9; 8, 7; 6, 5; 4, 3; 2, 1; ...
    17, 18; 19, 20; 21, 22; 23, 24];

prefixList = {'','','','','','','','','','CH','','','','','','CH','','','','','','','','','',''};
postfixList = {'','','','','','','','','','','','','','','','','','','','','','','','','',''};
myoChanList = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1:16, 0, 0, 0, 1:32, 1:16, {1:16, 17:32}, 0, 0, 0, 0, 0, {1:32, 33:64}, {1:32, 33:48}, ...
    '',''};
analogChanList = {'','','','','','','','','','CH17','','','','33','17','ADC1','','','','','','65','49','',''};
fileType = 'OE';

pathToYourConfigFile = '/media/jonathan/Data/Dropbox/mFiles/NPix/';

pList = {'Neuropixel-122321/122321_g0/122321_g0_imec0/', ...
    'Neuropixel-122421/122421_g0/122421_g0_imec0/', ...
    'Neuropixel-010522/010522_g0/010522_g0_imec0/', ...
    'Neuropixel-010622/010622_g0/010622_g0_imec0/', ...
    'Neuropixel-011322/011322_g0/011322_g0_imec0/', ...
    'Neuropixel-011422/011422_g0/011422_g0_imec0/', ...
    'Neuropixel-011822/011822_g0/011822_g0_imec0/', ...
    'Neuropixel-012622/012622_g0/012622_g0_imec0/', ...
    'Neuropixel-012722/012722_g0/012722_g0_imec0/', ...
    'Neuropixel-012822/012822_g0/012822_g0_imec0/', ...
    'Neuropixel-020122/020122_g0/020122_g0_imec0/', ...
    'Neuropixel-020222/020222_g0/020222_g0_imec0/', ...
    'Neuropixel-020322/020322_g0/020322_g0_imec0/', ...
    '', ...
    'Neuropixel-021722/021722_g0/021722_g0_imec0/', ...
    'Neuropixel-021822/021822_g0/021822_g0_imec0/', ...
    'Neuropixel-022522/022522_g0/022522_g0_imec0/', ...
    'Neuropixel-030422/030422_g0/030422_g0_imec0/' ...
    'Neuropixel-040622/040622_g0/040622_g0_imec0/' ...
    'Neuropixel-040722/040722_g0/040722_g0_imec0/' ...
    'Neuropixel-040822/040822_g0/040822_g0_imec0/' ...
    '', ...
    '', ...
    'Neuropixel-042022/042022_g0/042022_g0_imec0/', ...
    'Neuropixel-042222/042222_g0/042222_g0_imec0/'};

myoList = {'','','','','','','','','',...
    'Neuropixel-012822/2022-01-28_10-06-43/Record Node 102/', ...
    '','','','Neuropixel-020422/2022-02-04_09-32-48_corrected/Record Node 102/','Neuropixel-021722/2022-02-17_09-59-41/Record Node 102/',...
    {'Neuropixel-021822/2022-02-18_09-42-32_corr/Record Node 102/','Neuropixel-021822/2022-02-18_09-42-32_corr/Record Node 102/'},...
    '','','','','',{'Neuropixel-041422/2022-04-14_09-48-02/Record Node 102/', 'Neuropixel-041422/2022-04-14_09-48-02/Record Node 102/'}, ...
    {'Myomatrix_041422/Record Node 102/', 'Myomatrix_041422/Record Node 102/'}, '',''};
brokenChanList = {'','','','','','','','','',...
    [1 5 7:16],'','','',[],[],{[11:13],[]},'','','','','',{[1:11 13:27 29:32],[1:3 16:17]},...
    {[],[]},'',''};

loadDirs = {...
    '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', ...
    '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', ...
    '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', ...
    '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/SlowData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', ...
    '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', ...
    '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', ...
    '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/FastData/neuropixels/Data', ...
    '/media/jonathan/FastData/neuropixels/Data', '/media/jonathan/Data/Dropbox/KINARM', '/media/jonathan/FastData/neuropixels/Data',...
    '/media/jonathan/FastData/neuropixels/Data'};

for subject = 25
    for type = 1
        if type == 1
            thisList = pList{subject};
            phyDir = 'phyData';
            chanMapFile = '/media/jonathan/Data/Dropbox/mFiles/MATLABTOOLBOXES/Kilosort-main/configFiles/neuropixPhase3B1_kilosortChanMap.mat';
        else
            thisList = myoList{subject};
        end
        chanList = myoChanList{subject};
        if iscell(thisList)
            iters = length(thisList);
        else
            iters = 1;
        end
        for rr = 1:iters
            if iscell(thisList)
                innerList = thisList{rr};
                innerChanList = chanList{rr};
                innerBrokenList = brokenChanList{subject}{rr};
            else
                innerList = thisList;
                innerChanList = chanList;
                innerBrokenList = brokenChanList{subject};
            end
            if type == 2
                if length(innerChanList) == 16
                    chanMapFile = '/media/jonathan/Data/Dropbox/mFiles/NPix/MyoPixels/bipolar_test_kilosortChanMap.mat';
                elseif length(innerChanList) == 32
                    chanMapFile = '/media/jonathan/Data/Dropbox/mFiles/NPix/MyoPixels/monopolar_test_kilosortChanMap.mat';
                else
                    error('What.')
                end
                disp(['Using this channel map: ' chanMapFile])
                if rr == 1
                    phyDir = 'phyDataMyo';
                else
                    phyDir = ['phyDataMyo' num2str(rr)];
                end
            end
            
            if ~isempty(innerList)
                rootZ = [loadDirs{subject} '/' innerList];
                rootH = [rootZ phyDir '/']; % path to temporary binary file (same size as data, should be on fast SSD)
                if exist(rootH, 'dir')
                    eval(['!rm -rf ' rootH])
                    %error('We aleady have phyData')
                end
                mkdir(rootH)
                
                if subject == 15
                    ops.trange = [0 3050]; % time range to sort
                elseif subject == 16
                    ops.trange = [0 2132]; % computer crashed at this point
                else
                    ops.trange = [0 Inf];
                end
                ops.NchanTOT  = 385; % total number of channels in your recording
                
                run(fullfile(pathToYourConfigFile, 'NPix_config_3.m'))
                ops.fproc   = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
                ops.chanMap = fullfile(chanMapFile);

                           


                
                if rr == 1
                    checkFile = [rootZ 'MyomatrixData.bin'];
                else
                    checkFile = [rootZ 'MyomatrixData' num2str(rr) '.bin'];
                end
                if type == 2 && ~isfile(checkFile)
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
                    if rr == 1
                        fileID = fopen([rootZ 'MyomatrixData.bin'], 'w');
                    else
                        fileID = fopen([rootZ 'MyomatrixData' num2str(rr) '.bin'], 'w');
                    end
                    fwrite(fileID, dataSave, 'int16');
                    fclose(fileID);
                    save([rootZ 'syncData'], 'sync')
                    clear data dataSave
                end
                
                % find the binary file
                if type == 1
                    fs          = dir(fullfile([rootZ 'NeuropixelsRegistration/registered/'], '*.bin'));
                    ops.fbinary = fullfile([rootZ 'NeuropixelsRegistration/registered/'], fs(1).name);
                    %fs          = dir(fullfile(rootZ, '*tcat.imec0.ap.bin'));
                    %ops.fbinary = fullfile(rootZ, fs(1).name);
                    channelSep = 100; % Default for Neuropixels
                    overlap_s = 5e-4; % 5e-4 is default for Neuropixels
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


                    overlap_s = 0e-4; %2e-3 best
                    ops.sigmaMask = 1200; % gotta make me big
                end
                disp(ops.fbinary)
                
                rez                = preprocessDataSub(ops);
                rez                = datashift2(rez, 1);
                [rez, st3, tF]     = extract_spikes(rez);
                rez                = template_learning(rez, tF, st3);
                [rez, st3, tF]     = trackAndSort(rez);
                rez                = final_clustering(rez, tF, st3);
                
                ind = find(rez.st3(:,2) == 0);
                rez.st3(ind,:) = [];
                rez.xy(ind,:) = [];
                disp([num2str(length(ind)) ' deleted'])
                
                rez = remove_ks2_duplicate_spikes(rez, 'overlap_s', overlap_s, 'channel_separation_um', channelSep);
                
                %origRez = rez;
                %rez = origRez;
                rez                = find_merges(rez, 1);
                
                fprintf('found %d good units \n', sum(rez.good>0))
                if rr == 1
                    rootQ = fullfile(rootZ, 'kilosort3');
                else
                    rootQ = fullfile(rootZ, ['kilosort3-' num2str(rr)]);
                end
                %    eval(['!rm -rf ' rootQ])
                mkdir(rootQ)
                cd(rootQ)
                fprintf('Saving results to Phy  \n')
                rezToPhy2(rez, pwd);
            end
        end
    end
end
