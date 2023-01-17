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
mkdir([myomatrix '/sorted' num2str(myomatrix_num) '/']);

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
S = zeros(size(data,2), 2);
bipolarThresh = 90;
unipolarThresh = 120;
bipolar = length(chanList) == 16;
for q = 1:2
    if q == 1
        [b, a] = butter(2, [350 7500] / (30000/2), 'bandpass');
    elseif q == 2
        [b, a] = butter(2, [5 70] / (30000/2), 'bandpass');
    end
    useSeconds = 600;
    if size(data,1) < useSeconds*2*30000
        useSeconds = floor(size(data,1)/30000/2)-1;
    end
    tRange = size(data,1) - round(size(data,1)/2) - round(30000*useSeconds/2) : size(data,1) ...
             - round(size(data,1)/2) + round(30000*useSeconds/2);
    data_filt = zeros(length(tRange),size(data,2),'single');
    for i = 1:size(data,2)
        data_filt(:,i) = single(filtfilt(b, a, double(data(tRange,i))));
    end

    if q == 2
        S(:,q) = std(data_filt,[],1);
    else
        data_norm = data_filt ./ repmat(std(data_filt,[],1), [size(data_filt,1) 1]);
        spk = sum(data_norm < -8, 1);
        S(:,q) = spk / size(data_norm,1) * 30000;
    end

    subplot(1,2,q)
    if q == 1
        title('Filtered Signal Snippet (300-7500Hz)')
    else
        title('Filtered Noise Snippet (5-70Hz)')
    end
    hold on
    for i = 1:size(data,2)
        cmap = [0 0 0];
        if q == 1
            if S(i,1) < 1
                 cmap = [1 0.2 0.2];
            end
        else
            if (bipolar && S(i,2) > bipolarThresh) || (~bipolar && S(i,2) > unipolarThresh)
                cmap = [1 0.2 0.2];
            end
        end
        plot(data_filt(:,i) + i*1600, 'Color', cmap)
    end
    set(gca, 'YTick', (1:size(data,2))*1600, 'YTickLabels', 1:size(data,2))
    axis([1 size(data_filt,1) 0 (size(data,2)+1)*1600])
end
print([myomatrix '/sorted' num2str(myomatrix_num) '/brokenChan.png'], '-dpng')
S

if length(chanList) == 16
    brokenChan = find(S(:,2) > bipolarThresh | S(:,1) < 1);
else
    brokenChan = find(S(:,2) > unipolarThresh | S(:,1) < 1);
end
disp(['Broken/inactive channels are: ' num2str(brokenChan')])
save([myomatrix '/sorted' num2str(myomatrix_num) '/brokenChan.mat'], 'brokenChan');
clear data_filt data_norm

fileID = fopen([myomatrix '/sorted' num2str(myomatrix_num) '/data.bin'], 'w');
if true
    mean_data = mean(data,1);
    [b, a] = butter(4, [350 7500]/ (30000/2), 'bandpass');
    intervals = round(linspace(1, size(data,1), round(size(data,1)/(30000*60))));
    buffer = 256;
    for t = 1:length(intervals)-1
          preBuff = buffer; postBuff = buffer;
        if t == 1
            preBuff = 0;
        elseif t == length(intervals)-1
            postBuff = 0;
        end
        tRange = intervals(t)-preBuff : intervals(t+1)+postBuff;
        fdata = double(data(tRange,:)) - mean_data;
        fdata = fdata - median(fdata,2);
        fdata = filtfilt(b, a, fdata);
        fdata = fdata(preBuff+1 : end-postBuff-1, :);
        fdata(:,brokenChan) = 0;
        fwrite(fileID, int16(fdata'), 'int16');
    end
else
    fwrite(fileID, int16(data'), 'int16');
end
fclose(fileID);

if false
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
    save([myomatrix '/sorted' num2str(myomatrix_num) '/bulkEMG'], 'bEMG', 'notBroken', 'dataChan')
    clear bEMG
    disp('Saved generated bulk EMG')
end
disp('Saved myomatrix data binary')
quit