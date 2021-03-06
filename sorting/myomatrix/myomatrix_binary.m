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

if trange(2) == 0
    ops.trange = [0 Inf];
else
    ops.trange = trange;
end

dataChan = chanList;
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
if length(dataChan) == 32
    data = data(:,channelRemap);
end
ff = dir([myomatrix_data '/100_*' num2str(sync_chan) '.continuous']);
analogData = load_open_ephys_data([ff(1).folder '/' ff(1).name]);
analogData(analogData > 5) = 5;
sync = logical(round(analogData / max(analogData)));
clear analogData

save([myomatrix '/sync'], 'sync')
clear sync
disp('Saved sync data')

clf
for q = 1:2
    if q == 1
        [b, a] = butter(2, [350 7500] / (30000/2), 'bandpass');
    elseif q == 2
        [b, a] = butter(2, [10 200] / (30000/2), 'bandpass');
    end
    tRange = size(data,1) - (30000*60*40) : size(data,1) - (30000*60*30);
    data_filt = zeros(length(tRange),size(data,2),'single');
    for i = 1:size(data,2)
        data_filt(:,i) = single(filtfilt(b, a, double(data(tRange,i))));
    end
    subplot(1,2,q)
    hold on
    for i = 1:size(data,2)
        plot(data_filt(:,i) + i*2000)
    end
end
print([myomatrix '/brokenchan.png'], '-dpng')
S = std(data_filt,[],1);
if length(dataChan) == 32
    brokenChan = find(S > 80);
elseif length(dataChan) == 16
    brokenChan = find(S > 10);
end
S
disp('Broken channels are:')
brokenChan
data(:,brokenChan) = randn(size(data,1), length(brokenChan)) * 3e-1;
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
save([myomatrix '/bulkEMG'], 'bEMG', 'notBroken', 'dataChan')
clear bEMG
disp('Saved generated bulk EMG')
fileID = fopen([myomatrix '/data' num2str(myomatrix_num) '.bin'], 'w');
fwrite(fileID, int16(data'), 'int16');
fclose(fileID);
clear data
disp('Saved myomatrix data binary')
quit