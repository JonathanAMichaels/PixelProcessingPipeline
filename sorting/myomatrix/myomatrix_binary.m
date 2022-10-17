load('/tmp/config.mat')

% for use with monopolar
channelRemap = [23:-1:8 24:31 0:7] + 1;
% for use with bipolar
channelLabelsBipolar = [25, 26; 27, 28; 29, 30; 31, 32; ...
    16, 15; 14, 13; 12, 11; 10, 9; 8, 7; 6, 5; 4, 3; 2, 1; ...
    17, 18; 19, 20; 21, 22; 23, 24];

chanList = chans(1) : chans(2);
if length(chanList) == 16
    chanMapFile = [script_dir '/geometries/bipolar_test_kilosortChanMap.mat'];
elseif length(chanList) == 32
    chanMapFile = [script_dir '/geometries/monopolar_test_kilosortChanMap.mat'];
else
    error('Channel map not implemented')
end
disp(['Using this channel map: ' chanMapFile])

dataChan = chanList;

% Check if we're dealing with .dat or .continuous
oebin = dir([myomatrix_data '/structure.oebin']);
if isempty(oebin)
    ff = dir([myomatrix_data '/100_CH*' num2str(dataChan(1)) '.continuous']);
    if isempty(ff)
        prefix = [];
    else
        prefix = 'CH';
    end
    tempdata = load_open_ephys_data([myomatrix_data '/100_' prefix num2str(dataChan(1)) '.continuous']);
    tL = length(tempdata);
    clear tempdata
    data = zeros(tL, length(dataChan), 'int16');
    for chan = 1:length(dataChan)
        data(:,chan) = load_open_ephys_data([myomatrix_data '/100_' prefix num2str(dataChan(chan)) '.continuous']);
    end
    ff = dir([myomatrix_data '/100_*' num2str(sync_chan) '.continuous']);
    analogData = load_open_ephys_data([ff(1).folder '/' ff(1).name]);
else
    tempdata = load_open_ephys_binary([oebin(1).folder '/' oebin(1).name], 'continuous', 1, 'mmap');
    %data = zeros(size(tempdata.Data,2), length(dataChan), 'int16');
    if trange(2) == 0
        ops.trange = [1 size(tempdata.Data.Data(1).mapped,2)];
    else
        ops.trange = trange*30000 + 1;
    end
    data = tempdata.Data.Data(1).mapped(dataChan,ops.trange(1):ops.trange(2))';
    analogData = tempdata.Data.Data(1).mapped(sync_chan,ops.trange(1):ops.trange(2))';
    analogData(analogData < 10000) = 0;
    analodData(analogData >= 10000) = 1;
    clear tempdata
end

if length(dataChan) == 32
    data = data(:,channelRemap);
end

analogData(analogData > 5) = 5;
sync = logical(round(analogData / max(analogData)));
clear analogData

save([myomatrix '/sync'], 'sync')
clear sync
disp('Saved sync data')

disp(['Total recording time: ' num2str(size(data,1)/30000/60) ' minutes'])

clf
S = zeros(size(data,2), 3);
for q = 1:3
    if q == 1
        [b, a] = butter(2, [300 7500] / (30000/2), 'bandpass');
    elseif q == 2
        [b, a] = butter(2, [8000 14000] / (30000/2), 'bandpass');
    elseif q == 3
        [b, a] = butter(2, [10 60] / (30000/2), 'bandpass');
    end
    useSeconds = 30;
    tRange = size(data,1) - (30000*(120+useSeconds)) : size(data,1) - (30000*120);
    if isempty(tRange)
        tRange = -1;
    end
    while (tRange(1) < 1)
        useSeconds = useSeconds - 1;
        tRange = size(data,1) - (30000*useSeconds) : size(data,1);
    end

    data_filt = zeros(length(tRange),size(data,2),'single');
    for i = 1:size(data,2)
        data_filt(:,i) = single(filtfilt(b, a, double(data(tRange,i))));
    end
    subplot(1,3,q)
    hold on
    for i = 1:size(data,2)
        if size(data_filt,1) < 120000
            plot_range = 1:size(data_filt,1);
        else
            plot_range = 1:120000;
        end
        plot(data_filt(plot_range,i) + i*1500)
    end
    S(:,q) = std(data_filt,[],1);
end
print([myomatrix '/brokenchan' num2str(myomatrix_num) '.png'], '-dpng')

S

if length(dataChan) == 32
    brokenChan = find(S(:,2) > 16 | S(:,3) > 100);
elseif length(dataChan) == 16
    brokenChan = find(S(:,2) > 16 | S(:,3) > 20);
end
disp('Broken channels are:')
brokenChan
data(:,brokenChan) = randn(size(data,1), length(brokenChan))*3e-1;
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
save([myomatrix '/bulkEMG' num2str(myomatrix_num)], 'bEMG', 'notBroken', 'dataChan')
clear bEMG
disp('Saved generated bulk EMG')
fileID = fopen([myomatrix '/data' num2str(myomatrix_num) '.bin'], 'w');
fwrite(fileID, int16(data'), 'int16');
fclose(fileID);
clear data
disp('Saved myomatrix data binary')
quit