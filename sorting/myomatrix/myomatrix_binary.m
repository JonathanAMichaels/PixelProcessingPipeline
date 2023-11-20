script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'))

% for use with monopolar
channelRemap = [23:-1:8 24:31 0:7] + 1;
% for use with bipolar
% channelLabelsBipolar = [25, 26; 27, 28; 29, 30; 31, 32; ...
%     16, 15; 14, 13; 12, 11; 10, 9; 8, 7; 6, 5; 4, 3; 2, 1; ...
%     17, 18; 19, 20; 21, 22; 23, 24];

chanList = chans(1):chans(2);
disp(['Starting with these channels: ' num2str(chanList)])
disp(['Using this channel map: ' myo_chan_map_file])

dataChan = chanList;
if not(isfolder([myo_sorted_dir '/']))
    mkdir([myo_sorted_dir '/']);
end

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
        data(:, chan) = load_open_ephys_data([myomatrix_data '/100_' prefix num2str(dataChan(chan)) '.continuous']);
    end
    ff = dir([myomatrix_data '/100_*' num2str(sync_chan) '.continuous']);
    analogData = load_open_ephys_data([ff(1).folder '/' ff(1).name]);
else
    tempdata = load_open_ephys_binary([oebin(1).folder '/' oebin(1).name], 'continuous', 1, 'mmap');
    %data = zeros(size(tempdata.Data,2), length(dataChan), 'int16');
    if trange(2) == 0
        ops.trange = [1 size(tempdata.Data.Data(1).mapped, 2)];
    else
        ops.trange = trange * myo_data_sampling_rate + 1;
    end
    data = tempdata.Data.Data(1).mapped(dataChan, ops.trange(1):ops.trange(2))';
    try
        analogData = tempdata.Data.Data(1).mapped(sync_chan, ops.trange(1):ops.trange(2))';
    catch ME % to avoid "Index in position 1 exceeds array bounds (must not exceed XX)."
        if strcmp(ME.identifier, 'MATLAB:badsubscript')
            disp("No sync channel found, cannot save sync data")
            analogData = [];
        else
            rethrow(ME)
        end
    end
    if ~isempty(analogData)
        analogData(analogData < 10000) = 0;
        analogData(analogData >= 10000) = 1;
    end
    clear tempdata
end

if length(dataChan) == 32
    data = data(:, channelRemap);
end
if ~isempty(analogData)    
    analogData(analogData > 5) = 5;
    sync = logical(round(analogData / max(analogData)));
    clear analogData

    save([myomatrix '/sync'], 'sync')
    clear sync
    disp('Saved sync data')
end

disp(['Total recording time: ' num2str(size(data, 1) / myo_data_sampling_rate / 60) ' minutes'])

clf
S = zeros(size(data, 2), 3);
bipolarThresh = 90;
unipolarThresh = 120;
lowThresh = 0.1;
% bipolar = length(chanList) == 16;
% when q is 1, we will compute count the number of spikes in the channel and compare to a threshold
% when q is 2, we will compute the std of the low freq noise in the channel
% when q is 3, we will compute the SNR of the channel
for q = 1:4
    if q == 1
        [b, a] = butter(2, [250 4400] / (myo_data_sampling_rate / 2), 'bandpass');
    elseif q == 2
        [b, a] = butter(2, [5 100] / (myo_data_sampling_rate / 2), 'bandpass');
    elseif q == 3
        [b, a] = butter(2, 10000 / (myo_data_sampling_rate / 2), 'high');
    elseif q == 4
        [b, a] = butter(2, [300 1000] / (myo_data_sampling_rate / 2), 'bandpass');
    end
    useSeconds = 600;
    if size(data, 1) < useSeconds * 2 * myo_data_sampling_rate
        useSeconds = floor(size(data, 1) / myo_data_sampling_rate / 2) - 1;
    end
    tRange = size(data, 1) - round(size(data, 1) / 2) - round(myo_data_sampling_rate * useSeconds / 2):size(data, 1) ...
        - round(size(data, 1) / 2) + round(myo_data_sampling_rate * useSeconds / 2);
    data_norm = zeros(length(tRange), size(data, 2), 'single');
    data_filt = zeros(length(tRange), size(data, 2), 'single');
    for i = 1:size(data, 2)
        % standardize this data channel before filtering, but make sure not to divide by zero
        chan_std = std(single(data(tRange, i)));
        if chan_std == 0
            data_norm(:, i) = single(data(tRange, i));
        else
            data_norm(:, i) = single(data(tRange, i)) ./ chan_std;
        end
        % data_norm(:, i) = single(data(tRange, i)) ./ std(single(data(tRange, i)));
        % filter this data channel
        data_filt(:, i) = single(filtfilt(b, a, double(data_norm(:, i))));
    end

    if q == 1
        % normalize channels by std
        data_filt_norm = data_filt ./ repmat(std(data_filt, [], 1), [size(data_filt, 1) 1]);
        spk = sum(data_filt_norm < -7, 1); % check for spikes crossing 7 std below mean
        S(:, q) = spk / size(data_filt_norm, 1) * myo_data_sampling_rate;
    elseif q == 2
        S(:, q) = std(data_filt, [], 1); % get the std of the low freq noise
        % data_filt_norm = data_filt ./ repmat(S(:, q)', [size(data_filt, 1) 1]); % standardize
        low_band_power = rms(data_filt, 1) .^ 2;
    elseif q == 3
        S(:, q) = std(data_filt, [], 1); % get the std of the high freq noise
        % data_filt_norm = data_filt ./ repmat(S(:, q)', [size(data_filt, 1) 1]); % standardize
        high_band_power = rms(data_filt, 1) .^ 2;
    elseif q == 4
        % data_filt_norm = data_filt ./ repmat(std(data_filt, [], 1), [size(data_filt, 1) 1]); % standardize
        spike_band_power = rms(data_filt, 1) .^ 2;
        SNR = spike_band_power ./ (low_band_power + high_band_power);
        % replace any NaNs with 0
        SNR(isnan(SNR)) = 0;
        [~, idx] = sort(SNR, 'ascend');
        mean_SNR = mean(SNR);
        std_SNR = std(SNR);
        median_SNR = median(SNR);
        % get a MAD value for each channel
        MAD = median(abs(data_filt-mean(data_filt, 1)), 1);
        Gaussian_STDs = MAD / 0.6745;
        disp("Gaussian STDs: " + num2str(Gaussian_STDs))
        if isa(remove_bad_myo_chans, "char")
            rejection_criteria = remove_bad_myo_chans;
        else
            rejection_criteria = 'median';
        end
        disp("Using " + rejection_criteria + " threshold for SNR rejection criteria")

        % check by what criteria we should reject channels
        if strcmp(rejection_criteria, 'median')
            % reject channels with SNR < median
            SNR_reject_chans = chanList(SNR < median_SNR);
        elseif strcmp(rejection_criteria, 'mean')
            % reject channels with SNR < mean
            SNR_reject_chans = chanList(SNR < mean_SNR);
        elseif strcmp(rejection_criteria, 'mean-1std')
            % reject channels with SNR < mean - std
            SNR_reject_chans = chanList(SNR < mean_SNR - std_SNR);
        elseif startsWith(rejection_criteria, 'percentile')
            % ensure that the percentile is numeric and between 0 and 100
            percentile = str2double(rejection_criteria(11:end));
            if isnan(percentile) || percentile < 0 || percentile > 100
                error("Error with 'remove_bad_myo_chans' setting in config.yaml. Numeric value after 'percentile' must be between 0 and 100")
            end
            % reject channels with SNR < Nth percentile
            percentile_SNR = prctile(SNR, percentile);
            SNR_reject_chans = chanList(SNR < percentile_SNR);
        elseif startsWith(rejection_criteria, 'lowest')
            % ensure that the number of channels to reject is numeric and less than the number of channels
            N_reject = str2double(rejection_criteria(7:end));
            if isnan(N_reject) || N_reject < 0 || N_reject > length(chanList)
                error("Error with 'remove_bad_myo_chans' setting in config.yaml. Numeric value after 'lowest' must be between 0 and " + length(chanList))
            end
            % reject N_reject lowest SNR channels
            SNR_reject_chans = chanList(idx(1:N_reject));
        end

        % [~, idx] = sort(SNR, 'ascend');
        % idx = idx(1:floor(length(idx) / 2));
        % bitmask = zeros(length(chanList), 1);
        % bitmask(idx) = 1;
        % SNR_reject_chans = chanList(bitmask == 1);
        disp("SNRs: " + num2str(SNR))
        disp("Mean +/- Std. SNR: " + num2str(mean_SNR) + " +/- " + num2str(std_SNR))
        disp("Median SNR: " + num2str(median_SNR))
        disp("Channels with rejectable SNRs: " + num2str(SNR_reject_chans))
    end

    % subplot(1, 4, q)
    % if q == 1
    %     title('Filtered Signal Snippet (250-4400Hz)')
    % elseif q == 2
    %     title('Filtered Noise Snippet (5-70Hz)')
    % end
    % hold on
    % for i = 1:size(data, 2)
    %     cmap = [0 0 0];
    %     if q == 1
    %         if S(i, 1) < lowThresh
    %             cmap = [1 0.2 0.2];
    %         end
    %     elseif q == 2
    %         if (bipolar && S(i, 2) > bipolarThresh) || (~bipolar && S(i, 2) > unipolarThresh)
    %             cmap = [1 0.2 0.2];
    %         end
    %     end
    %     plot(data_filt(:, i) + i * 1600, 'Color', cmap)
    % end
    % set(gca, 'YTick', (1:size(data, 2)) * 1600, 'YTickLabels', 1:size(data, 2))
    % axis([1 size(data_filt, 1) 0 (size(data, 2) + 1) * 1600])
end
print([myo_sorted_dir '/brokenChan.png'], '-dpng')

if length(chanList) == 16
    % check for broken channels if meeting various criteria, including: high std, low spike rate, low SNR. Eliminate if any true
    brokenChan = int64(union(find(S(:, 2) > bipolarThresh | S(:, 1) < lowThresh), SNR_reject_chans)); %S(:, 3) > bipolarThresh
else
    brokenChan = int64(union(find(S(:, 2) > unipolarThresh | S(:, 1) < lowThresh), SNR_reject_chans)); %S(:, 3) > unipolarThresh
end
disp(['Automatically detected rejectable channels are: ' num2str(brokenChan')])

% now actually remove the detected broken channels if True
% if a list of broken channels is provided, use that instead
% if false, just continue
if isa(remove_bad_myo_chans(1), 'logical') || isa(remove_bad_myo_chans, 'char')
    if remove_bad_myo_chans(1) == false
        brokenChan = [];
        disp('Not removing any broken/noisy channels, because remove_bad_myo_chans is false')
        % disp(['Keeping channel list: ' num2str(chanList)])
    elseif remove_bad_myo_chans(1) == true || isa(remove_bad_myo_chans, 'char')
        data(:, brokenChan) = [];
        chanList(brokenChan) = [];
        disp(['Just removed automatically detected broken/noisy channels: ' num2str(brokenChan')])
        disp(['New channel list is: ' num2str(chanList)])
    end
elseif isa(remove_bad_myo_chans, 'integer')
    brokenChan = remove_bad_myo_chans; % overwrite brokenChan with manually provided list
    data(:, brokenChan) = [];
    chanList(brokenChan) = [];
    disp(['Just removed manually provided broken/noisy channels: ' num2str(brokenChan)])
    disp(['New channel list is: ' num2str(chanList)])
else
    error('remove_bad_myo_chans must be a boolean, string with SNR rejection method, or an integer list of channels to remove')
end

save([myo_sorted_dir '/chanList.mat'], 'chanList')
save([myo_sorted_dir '/brokenChan.mat'], 'brokenChan');

% load and modify channel map variables to remove broken channel elements, if desired
if ~isempty(brokenChan) && remove_bad_myo_chans(1) ~= false
    load(myo_chan_map_file)
    % if size(data, 2) >= num_KS_components
    %     chanMap(brokenChan) = []; % take off end to save indexing
    %     chanMap0ind(brokenChan) = []; % take off end to save indexing
    %     connected(brokenChan) = [];
    %     kcoords(brokenChan) = [];
    %     xcoords(brokenChan) = [];
    %     ycoords(brokenChan) = [];
    % else
    numDummy = max(0, num_KS_components - size(data, 2)); % make sure it's not negative
    dummyData = zeros(size(data, 1), numDummy, 'int16');
    data = [data dummyData]; % add dummy channels to make size larger than num_KS_components
    chanMap = 1:size(data, 2);
    chanMap0ind = chanMap - 1;
    connected = true(size(data, 2), 1);
    kcoords = ones(size(data, 2), 1);
    xcoords = zeros(size(data, 2), 1);
    ycoords = (size(data, 2):-1:1)';
    % end
    disp('Broken channels were just removed from that channel map')
    save(fullfile(myo_sorted_dir, 'chanMapAdjusted.mat'), 'chanMap', 'connected', 'xcoords', ...
        'ycoords', 'kcoords', 'chanMap0ind', 'fs', 'name', 'numDummy', 'Gaussian_STDs')
else
    copyfile(myo_chan_map_file, fullfile(myo_sorted_dir, 'chanMapAdjusted.mat'))
    % add numDummy to chanMapAdjusted.mat
    load(fullfile(myo_sorted_dir, 'chanMapAdjusted.mat'))
    numDummy = 0;
    save(fullfile(myo_sorted_dir, 'chanMapAdjusted.mat'), 'chanMap', 'connected', 'xcoords', ...
        'ycoords', 'kcoords', 'chanMap0ind', 'fs', 'name', 'numDummy', 'Gaussian_STDs')
end

clear data_filt data_norm

fileID = fopen([myo_sorted_dir '/data.bin'], 'w');
% if true
disp("Filtering raw data with passband:")
disp(strcat(string(myo_data_passband(1)), "-", string(myo_data_passband(2)), " Hz"))
mean_data = mean(data, 1);
[b, a] = butter(4, myo_data_passband / (myo_data_sampling_rate / 2), 'bandpass');
intervals = round(linspace(1, size(data, 1), round(size(data, 1) / (myo_data_sampling_rate * 5))));
if numDummy > 0
    chanIdxsToFilter = 1:num_KS_components-numDummy;
else
    chanIdxsToFilter = 1:size(data, 2);
end
buffer = 128;
% now write the data to binary file in chunks of 5 seconds, but exclude dummy channels
for t = 1:length(intervals) - 1
    preBuff = buffer; postBuff = buffer;
    if t == 1
        preBuff = 0;
    elseif t == length(intervals) - 1
        postBuff = 0;
    end
    tRange = intervals(t) - preBuff:intervals(t + 1) + postBuff;
    fdata = double(data(tRange, :)) - mean_data;
    fdata(:, chanIdxsToFilter) = fdata(:, chanIdxsToFilter) - median(fdata(:, chanIdxsToFilter), 1);
    fdata = filtfilt(b, a, fdata);
    fdata = fdata(preBuff + 1:end - postBuff - 1, :);
    % fdata(:, brokenChan) = randn(size(fdata(:, brokenChan))) * 5;
    fwrite(fileID, int16(fdata'), 'int16');
end
% else
%     data(:, brokenChan) = randn(size(data(:, brokenChan))) * 5;
%     fwrite(fileID, int16(data'), 'int16');
% end
fclose(fileID);
% if false
%     % Generate "Bulk EMG" dataset
%     notBroken = 1:size(data, 2);
%     notBroken(brokenChan) = [];
%     if length(dataChan) == 32
%         bottomHalf = [9:16 25:32];
%         topHalf = [1:8 17:24];
%         bottomHalf(ismember(bottomHalf, brokenChan)) = [];
%         topHalf(ismember(topHalf, brokenChan)) = [];
%         bEMG = int16(mean(data(:, bottomHalf), 2)) - int16(mean(data(:, topHalf), 2));
%     else
%         bEMG = int16(mean(data(:, notBroken), 2));
%     end
%     save([myo_sorted_dir '/bulkEMG'], 'bEMG', 'notBroken', 'dataChan')
%     clear bEMG
%     disp('Saved generated bulk EMG')
% end
disp('Saved myomatrix data binary')
quit
