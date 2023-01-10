% This function re-sorts kilosort output to merge single units that are
% time-shifted version of each other and select units with acceptable SNR.

% For Myomatrix data kilsort often produces units that are time-shifted versions of each other due to
% 1. the relative wideness of motor units waveforms relative to cortical waveforms,
% 2. the procedure of fitting templates to residuals of spikes,
% 3. the complex propogation of waveforms across channels that occurs as a results of spikes travelling along muscle fibers

% INPUTS: params struct, must include:
%   params.binaryFile: location of binary file created by kilosort
%   params.kiloDir: directory including kilosort outputs in .npy format
%   params.chanMap: (chan, 2) matrix of electrode spatial locations in um. This is only used for plotting.
%   
%   optional parameters:
%   params.sr: sampling rate (Hz)
%   params.userSorted: whether to load user-curated spike sorting or kilosort output
%   params.savePlot: whether or not to save waveform plots
%   params.crit: correlation threshold for merging clusters
%   params.SNRThresold: SNR threshold for including units at the final step
%   params.refractoryLim: inter-spike times below this threshold (in ms) will be eliminated as duplicate spikes
%   params.backSp: number of samples to extract before waveform peak
%   params.forwardSp: number of samples to extract after waveform peak
%   params.corrRange: range of sliding cross-correlation in samples
%   params.waveCount: maximum number of waveforms to extract per cluster

% OUTPUTS: custom_merge.mat file saved in kilosort directory with new clusters and mean waveforms
%   Optionally waveform plots are saved to kiloDir/Plots/
 
function Myomatrix_resorter(params)

xcoords = params.chanMap(:,1);
ycoords = params.chanMap(:,2);
if ~isfield(params, 'sr')
    params.sr = 30000;
end
if ~isfield(params, 'userSorted')
    params.userSorted = false;
end
if ~isfield(params, 'doPlots')
    params.doPlots = true;
end
if ~isfield(params, 'savePlots')
    params.savePlots = false;
end
% minimum correlation to be considered as originating from one cluster
if ~isfield(params, 'crit')
    params.crit = 0.75;
end
% SNR threshold for keeping clusters at the end
if ~isfield(params, 'SNRThreshold')
    params.SNRThreshold = 3.2;
end
if ~isfield(params, 'multiSNRThreshold')
    params.multiSNRThreshold = 3.8; % 3.8
end
if ~isfield(params, 'consistencyThreshold')
    params.consistencyThreshold = 0.7;
end
% Spikes below this refractory time limit will be considered duplicates
if ~isfield(params, 'refractoryLim')
    params.refractoryLim = 4;
end
% Define temporal sample range for waveforms (wider than kilosort!)
if ~isfield(params, 'backSp')
    params.backSp = round(params.sr * 0.0034);
end
if ~isfield(params, 'forwardSp')
    params.forwardSp = round(params.sr * 0.0034);
end
% Time range for cross-correlation
if ~isfield(params, 'corrRange')
    params.corrRange = floor((params.backSp + params.forwardSp) / 1.05);
end
% Max number of random spikes to extract per cluster
if ~isfield(params, 'waveCount')
    params.waveCount = 800;
end
if ~isfield(params, 'skipFilter')
    params.skipFilter = false;
end

dbstop if error

% Read data from kilosort output
disp('Reading kilosort output')
T = readNPY([params.kiloDir '/spike_times.npy']);
I = readNPY([params.kiloDir '/spike_clusters.npy']);
Wrot = readNPY([params.kiloDir '/whitening_mat_inv.npy']);
load([params.kiloDir '/brokenChan']);
params.brokenChan = brokenChan;

%TMP = readNPY([params.kiloDir '/templates.npy']);
%TMP_ind = readNPY([params.kiloDir '/templates_ind.npy']);

if params.userSorted
    clusterGroup = tdfread([params.kiloDir '/cluster_group.tsv']);
else
    clusterGroup = tdfread([params.kiloDir '/cluster_KSLabel.tsv']);
    clusterGroup.group = clusterGroup.KSLabel;
end
C = []; C_ident = [];
for i = 1:length(clusterGroup.cluster_id)
    sp = find(I == clusterGroup.cluster_id(i));
    C(end+1) = clusterGroup.cluster_id(i);
    C_ident(end+1) = strcmp(clusterGroup.group(i,1:3), 'goo');
end

if ~params.skipFilter
    % Extract individual waveforms from kilosort binary
    [mdata, data, consistency] = extractWaveforms(params, T, I, C, Wrot, true);

    if false
    % re-center all spike times
    temp = permute(mdata, [3 1 2]);
    [~, minTime] = min(min(temp,[],3),[],2);
    for j = 1:length(C)
        T(I == C(j)) = T(I == C(j)) + minTime(j) - params.backSp;
    end
    end

    % calc stats
    [SNR, spkCount] = calcStats(mdata, data, T, I, C);
    SNR

    % Kilosort is bad at selecting which motor units are 'good', since it uses ISI as a criteria. We expect many
    % spike times to be close together.
    % Take only 'good' single units with sufficient SNR
    C = C(C_ident == 1 | (SNR > params.multiSNRThreshold & spkCount > 50));
end

% Let's straight up trim off everything we don't need to save time
keepSpikes = find(ismember(I,C));
I = I(keepSpikes);
T = T(keepSpikes);

disp(['Number of clusters to work with: ' num2str(length(C))])
disp(['Number of spikes to work with: ' num2str(length(I))])

% Iteratively combine clusters that are similar to each other above some
% threshold
keepGoing = 1;
while keepGoing
    % Extract individual waveforms from kilosort binary
    [mdata, ~, consistency] = extractWaveforms(params, T, I, C, Wrot, true);

    % re-center all spike times
    if false
    temp = permute(mdata, [3 1 2]);
    [~, minTime] = min(min(temp,[],3),[],2);
    new_mdata = zeros(size(mdata,1)*3, size(mdata,2), size(mdata,3));
    for j = 1:length(C)
        T(I == C(j)) = T(I == C(j)) + minTime(j) - params.backSp;
        % correct mdata centering
        new_mdata((size(mdata,1)+1:size(mdata,1)*2) - (minTime(j) - params.backSp),:,j) = mdata(:,:,j);
    end
    mdata = new_mdata((size(mdata,1)+1:size(mdata,1)*2),:,:);
    end

    % calculate cross-correlation
    [bigR, lags, rCross] = calcCrossCorr(params, mdata, consistency, T, I, C);

    disp('Combining units and re-assigning spikes')
    % Find lags with maximum correlation
    [m, mL] = max(bigR,[],1);
    m = squeeze(m); mL = squeeze(mL);
    m(isnan(m)) = 0;
    mL = lags(mL);
    
    % Let's choose what to merge
    J = m > params.crit | (m > 0.6 & rCross > 0.3);
    
    % Create graph of connected clusters
    J = graph(J);
    bins = conncomp(J);
    figure(999)
    clf
    subplot(1,2,1)
    imagesc(m)
    colorbar
    subplot(1,2,2)
    hold on
    title('Graph of connected clusters')
    plot(J)
    axis off
    drawnow
    
    % Get minimum amplitudes of channel/cluster pair
    temp = permute(mdata, [3 1 2]);
    ampList = min(temp(:,:), [], 2);
    [~, minAmpList] = min(min(temp,[],3),[],2);
    
    % Shift spike times of a single cluster into frame of biggest amplitude
    % channel
    newLags = zeros(1,length(C));
    newC = bins;
    for j = 1:max(bins)
        ind = find(bins == j);
        [~, mi] = min(ampList(ind));
        if length(ind) > 1
            shiftInd = ind;
            shiftInd(mi) = [];
            newLags(shiftInd) = mL(ind(mi),shiftInd);
        end
        newLags(ind) = newLags(ind) + minAmpList(ind(mi)) - params.backSp;
    end
    
    % Adjust spike times and combine clusters
    newT = T;
    newI = I;
    for i = 1:length(T)
        ind = find(I(i) == C);
        if ~isempty(ind)
            newI(i) = newC(ind);
            newT(i) = T(i) + newLags(ind);
        end
    end
    T = newT;
    I = newI;
    C = unique(newC);

    % remove duplicates
    %[T, I] = removeDuplicates(params, T, I, C);

    % When there are no more connected clusters we can stop
    keepGoing = length(bins) ~= length(unique(bins));
end
disp('Finished merging clusters')

if false
% re-center all spike times
temp = permute(mdata, [3 1 2]);
[~, minTime] = min(min(temp,[],3),[],2);
for j = 1:length(C)
    T(I == C(j)) = T(I == C(j)) + minTime(j) - params.backSp;
end
end

% Re-extract
[mdata, data, consistency] = extractWaveforms(params, T, I, C, Wrot, true);
% use first vs last quartel as consistency check
RR = consistency.R;
RR(isnan(RR) | RR < 0) = 0;
disp('waveform consistency')
RR

% Re-calc stats
[SNR, spkCount] = calcStats(mdata, data, T, I, C);
disp('SNR')
SNR

% Remove clusters that don't meet inclusion criteria
saveUnits = find(SNR > params.SNRThreshold & spkCount > 50 & ...
    RR >= params.consistencyThreshold);
keepSpikes = find(ismember(I, saveUnits));
T = T(keepSpikes);
I = I(keepSpikes);
C = unique(I);
mdata = mdata(:,:,saveUnits);
data = data(:,:,:,saveUnits);
SNR = SNR(saveUnits);
spkCount = spkCount(saveUnits);
consistency.R = consistency.R(saveUnits);
consistency.wave = consistency.wave(:,:,:,saveUnits);
consistency.channel = consistency.channel(:,saveUnits);

disp(['Keeping ' num2str(length(C)) ' Units'])

if params.doPlots
    % Plot waveforms for each unit
    for j = 1:size(mdata,3)
        firstNan = find(isnan(squeeze(data(1,1,:,j))),1) - 1;
        if isempty(firstNan)
            firstNan = size(data,3);
        end
        temp = mdata(:,:,j);
        yScale = (max(temp(:))-min(temp(:)))/1500;
        figure(j)
        set(gcf, 'Position', [j*50 1 250 400])
        clf
        ttl = sprintf(['Spikes: ' num2str(spkCount(j)) '\nmax-SNR: ' num2str(SNR(j))]);
        title(ttl)
        hold on
        for e = 1:size(mdata,2)
            thisTemplate = mdata(:,e,j);
            plot((1:size(thisTemplate,1)) + xcoords(e)/2, ...
                thisTemplate + ycoords(e)*yScale, 'LineWidth', 1.5, 'Color', [0 0 0])%[0 0 0 0.015])
        end
        axis off
        inc = abs(mode(diff(ycoords)))*yScale;
        set(gca, 'YLim', [min(ycoords)*yScale-inc max(ycoords)*yScale+inc])
        if params.savePlots
            if ~exist([params.kiloDir '/Plots'], 'dir')
                mkdir([params.kiloDir '/Plots'])
            end
            print([params.kiloDir '/Plots/' num2str(j) '.png'], '-dpng')
            print([params.kiloDir '/Plots/' num2str(j) '.svg'], '-dsvg')
        end
    end

    % Plot average waveform from beginning and end of recording
    for j = 1:size(mdata,3)
        firstNan = find(isnan(squeeze(data(1,1,:,j))),1) - 1;
        if isempty(firstNan)
            firstNan = size(data,3);
        end
        if firstNan < 1000
            firstBunch = 1:round(firstNan/2);
            lastBunch = round(firstNan/2)+1:firstNan;
        else
            firstBunch = 1:500;
            lastBunch = firstNan-499:firstNan;
        end
        temp = mdata(:,:,j);
        yScale = (max(temp(:))-min(temp(:)))/1500;
        figure(j+100)
        set(gcf, 'Position', [j*50 1 250 400])
        clf
        ttl = sprintf(['Spikes: ' num2str(spkCount(j)) '\nmax-SNR: ' num2str(SNR(j))]);
        title(ttl)
        hold on
        for e = 1:size(mdata,2)
            thisTemplate = squeeze(mean(data(:,e,firstBunch,j),3));
            plot((1:size(thisTemplate,1)) + xcoords(e)/2, ...
                thisTemplate + ycoords(e)*yScale, 'LineWidth', 2, 'Color', [0 0 0.7 0.5])
            thisTemplate = squeeze(mean(data(:,e,lastBunch,j),3));
            plot((1:size(thisTemplate,1)) + xcoords(e)/2, ...
                thisTemplate + ycoords(e)*yScale, 'LineWidth', 2, 'Color', [0.7 0 0 0.5])
        end
        axis off
        inc = abs(mode(diff(ycoords)))*yScale;
        set(gca, 'YLim', [min(ycoords)*yScale-inc max(ycoords)*yScale+inc])
        if params.savePlots
            if ~exist([params.kiloDir '/Plots'], 'dir')
                mkdir([params.kiloDir '/Plots'])
            end
            print([params.kiloDir '/Plots/' num2str(j) '-wavecomp.png'], '-dpng')
            print([params.kiloDir '/Plots/' num2str(j) '-wavecomp.svg'], '-dsvg')
        end
    end

    % Plot histogram of inter-spike times
    figure(1000)
    clf
    for j = 1:length(C)
        subplot(ceil(sqrt(length(C))),ceil(sqrt(length(C))),j)
        times = T(I == C(j));
        dt = diff(times/(params.sr/1000));
        histogram(dt, 0:2:150, 'EdgeColor', 'none')
        box off
        xlabel('Inter-spike time (ms)')
        ylabel('Count')
    end
    if params.savePlots
        if ~exist([params.kiloDir '/Plots'], 'dir')
            mkdir([params.kiloDir '/Plots'])
        end
        print([params.kiloDir '/Plots/histogram.png'], '-dpng')
    end
end

disp(['Number of clusters: ' num2str(length(C))])
disp(['Number of spikes: ' num2str(length(I))])
save([params.kiloDir '/custom_merge.mat'], 'T', 'I', 'C', 'mdata', 'SNR', 'consistency');
end


function [SNR, spkCount] = calcStats(mdata, data, T, I, C)
disp('Calculating waveform stats')
spkCount = zeros(1,size(mdata,3));
SNR = zeros(1,size(mdata,3));
for j = 1:size(mdata,3)
    spkCount(j) = length(T(I == C(j)));
    firstNan = find(isnan(squeeze(data(1,1,:,j))),1) - 1;
    if isempty(firstNan)
        firstNan = size(data,3);
    end
    useSpikes = 1:firstNan;
    useData = squeeze(permute(data(:,:,useSpikes,j), [1 3 2]));
    mWave = repmat(permute(mdata(:,:,j),[1 3 2]), [1 size(useData,2) 1]);
    % calculate SNR
    tempSNR = squeeze(sum((max(useData,[],1) - min(useData,[],1)) ./ (2 * std(useData - mWave,[],1))) / size(useData,2));
    SNR(j) = max(tempSNR);
end
end

function [mdata, data, consistency] = extractWaveforms(params, T, I, C, Wrot, unwhiten)
disp('Extracting waveforms from binary')
f = fopen(params.binaryFile, 'r');
recordSize = 2; % 2 bytes for int16
nChan = size(params.chanMap,1);
spt = recordSize*nChan;
badChan = params.brokenChan; % Zero out channels that are bad
Wrot_orig = pinv(Wrot) * 200; % recover the original whitening matrix (specific to pykilosort 2.5)
totalT = double(max(T));
sections = linspace(1, totalT, 3); % split into 2 equal parts
waveParcel = floor(params.waveCount/(length(sections)-1));
% Extract each waveform
data = nan(params.backSp + params.forwardSp, nChan, params.waveCount, length(C), 'single');
mdata = zeros(params.backSp + params.forwardSp, nChan, length(C), 'single');
R = zeros(1,length(C),'single');
consistency = struct('R',[],'wave',[],'channel',[]);
for j = 1:length(C)
    disp(['Extracting unit ' num2str(j) ' of ' num2str(length(C))])
    tempdata = nan(params.backSp + params.forwardSp, nChan, waveParcel, length(sections)-1, 'single');
    waveStep = 0;
    for q = 1:(length(sections)-1)
        times = T(I == C(j));
        times = times(times >= sections(q) & times < sections(q+1)); % trim times
        innerWaveCount = min([waveParcel length(times)]);
        useTimes = times(round(linspace(1, length(times), innerWaveCount)));
        for t = 1:length(useTimes)
            fseek(f, (useTimes(t)-params.backSp) * spt, 'bof');
            tempdata(:,:,t,q) = fread(f, [nChan, params.backSp+params.forwardSp], '*int16')';
            if unwhiten
                tempdata(:,:,t,q) = tempdata(:,:,t,q) / Wrot_orig; % unwhiten and rescale data to uV
            end
            tempdata(:,badChan,t,q) = 0;
        end
        data(:,:,waveStep+1 : waveStep+innerWaveCount,j) = tempdata(:,:,1:innerWaveCount,q);
        waveStep = waveStep+innerWaveCount;
    end
    mdata(:,:,j) = nanmean(data(:,:,:,j),3);
    
    % consistency check
    if nChan >= 384
        grabChannels = 8;
    elseif nChan == 32
        grabChannels = 6;
    elseif nChan == 16
        grabChannels = 3;
    else
        grabChannels = 4;
    end
    tempm = squeeze(nanmean(tempdata,3));
    ucheck = permute(tempm, [2 1 3]);
    ucheck = ucheck(:,:);
    [~, ind] = sort(range(ucheck,2), 'descend');
    consistency.wave(:,:,:,j) = tempm(:,ind(1:grabChannels),:);
    consistency.channel(:,j) = ind(1:grabChannels);
    tempm = permute(tempm(:,ind(1:grabChannels),:), [3 1 2]);
    tempm = tempm(:,:)';
    tempCorr = corr(tempm);
    R(j) = tempCorr(1,2);
end
fclose(f);
consistency.R = R;
end

function [r, lags, rCross] = calcCrossCorr(params, mdata, consistency, T, I, C)
    disp('Calculating waveform cross-correlations')
    mdata = single(mdata);
    % Let's focus on the top channels only
    if size(mdata,2) > 32
        for j = 1:size(mdata,3)
            allChan = 1:size(mdata,2);
            allChan(consistency.channel(:,j)) = [];
            mdata(:,allChan,j) = 0;
        end
    end
    % concatenate channels together while keeping a buffer between them
    catdata = [];
    catdata = cat(1, catdata, zeros(params.corrRange, size(mdata,3)));
    for j = 1:size(mdata,2)
        catdata = cat(1, catdata, squeeze(mdata(:,j,:)));
        catdata = cat(1, catdata, zeros(params.corrRange+1, size(mdata,3)));
    end
    catdata = single(catdata);

    % xcorr can handle this without a for-loop, but it uses too much memory that way...
    count = 1;
    r = zeros(params.corrRange*2 + 1, size(catdata,2)^2, 'single');
    for i = 1:size(catdata,2)
        for j = 1:size(catdata,2)
            [r(:,count), lags] = xcorr(catdata(:,i), catdata(:,j), params.corrRange, 'normalized');
            count = count + 1;
        end
    end
    r = reshape(r, [size(r,1) size(mdata,3) size(mdata,3)]);
    for z = 1:size(r,1)
        r(z, logical(eye(size(r,2), size(r,3)))) = 0;
    end


    % Calculate zero-lag auto and cross-correlograms in spike timing (using a 5ms bin)
    M = ceil(double(max(T)/30/5));
    S = zeros(M,length(C),'logical');
    for j = 1:length(C)
        spk = round(double(T(I == C(j)))/30/5);
        S(spk,j) = 1;
    end
    rCross = zeros(size(S,2), size(S,2));
    for i = 1:size(S,2)
        for j = 1:size(S,2)
            rCross(i,j) = corr(S(:,i), S(:,j));
        end
    end
end


function [T, I] = removeDuplicates(params, T, I, C)
    % Let's remove spikes that were multi-detected
    sampThresh = params.refractoryLim * (params.sr/1000);
    disp(['Removing duplicate counted spikes within ' num2str(sampThresh/(params.sr/1000)) 'ms range'])
    for j = 1:length(C)        
        ind = find(I == C(j));
        times = T(ind);

        % This is the most efficient way to eliminate duplicate spikes.
        % You'll drop a few more than necessary, but it's generally worth it for the efficiency
        dt1 = diff(times);
        dt2 = flipud(diff(flipud(times)));
        delInd = find(dt1 <= sampThresh & dt2 <= sampThresh) + 1;
        I(ind(delInd)) = [];
        T(ind(delInd)) = [];

        ind = find(I == C(j));
        times = T(ind);
        keepRemoving = true;
        delInd = [];
        while keepRemoving
            theseTimes = times;
            theseTimes(delInd) = [];
            dt = diff(theseTimes);
            if sum(dt <= sampThresh) == 0
                keepRemoving = false;
            else
                delInd(end+1) = find(dt <= sampThresh, 1)  + 1 + length(delInd);
            end
        end
        I(ind(delInd)) = [];
        T(ind(delInd)) = [];
    end
end
