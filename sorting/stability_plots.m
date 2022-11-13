clear

try
    load('~/PixelProcessingPipeline/geometries/neuropixPhase3B1_kilosortChanMap');
catch
    load('/Users/jonathanamichaels/Library/CloudStorage/Dropbox/mFiles-Projects/GitHub/PixelProcessingPipeline/geometries/neuropixPhase3B1_kilosortChanMap')
end
T = readNPY('spike_times.npy');
I = readNPY('spike_clusters.npy');
TP = readNPY('templates.npy');
TP_ind = readNPY('spike_templates.npy')+1;
clusterGroup = tdfread('cluster_KSLabel.tsv');
C = [];
for j = 1:size(clusterGroup.cluster_id,1)
    if strcmp(clusterGroup.KSLabel(j,:), 'good')
        C(end+1) = clusterGroup.cluster_id(j);
    end
end
keepSpikes = ismember(I, C);
T = T(keepSpikes);
I = I(keepSpikes);
TP_ind = TP_ind(keepSpikes);
cluster_um = zeros(1,length(C));
for j = 1:length(C)
    tmp = squeeze(range(TP(TP_ind(find(I == C(j),1)),:,:),2));
    [~, temp] = max(tmp);
    cluster_um(j) = ycoords(temp);
end


f = fopen('proc.dat', 'r');
recordSize = 2; % 2 bytes for int16
nChan = 384;
spt = recordSize*nChan;


% gather SD per channel
times = round(linspace(1, double(max(T)), 30000));
binEdges = round(linspace(1, double(max(T)), 60));
samples = zeros(length(times),384);
for i = 1:length(times)
    fseek(f, times(i) * spt, 'bof');
    samples(i,:) = fread(f, [nChan, 1], '*int16');  
end
sd = std(double(samples),[],1);

sp = zeros(length(binEdges)-1,384);
for b = 1:length(binEdges)-1
    ind = find(times >= binEdges(b) & times < binEdges(b+1));
    sp(b,:) = std(samples(ind,:),[],1);
end


comChan = 3;
tmp_amp = zeros(1,length(T)); tmp_um = zeros(1,length(T));
for i = 1:length(T)
    if mod(i,100000) == 0
        disp(i)
    end
    fseek(f, T(i) * spt, 'bof');
    tmp = abs(double(fread(f, [nChan, 1], '*int16')))';
    tmp = tmp ./ sd;
    tmp(tmp < 4) = 0;
    [m, ind] = sort(tmp, 'descend');
    norm_chan = m(1:comChan) / sum(m(1:comChan));
    tmp_um(i) = ceil(sum(ycoords(ind(1:comChan)) .* norm_chan') / 10);
    tmp_amp(i) = mean(m(1:comChan));
end
fclose(f);
delInd = find(isnan(tmp_um));
tmp_um(delInd) = []; tmp_amp(delInd) = []; T(delInd) = []; I(delInd) = [];

Ts = round(double(T)/30000/5)+1;
totalT = max(Ts);
raster1 = zeros(totalT,max(tmp_um));
rasteramp = zeros(totalT,max(tmp_um));
for i = 1:length(T)
    raster1(Ts(i), tmp_um(i)) = ...
        raster1(Ts(i), tmp_um(i)) + 1;
    rasteramp(Ts(i), tmp_um(i)) = ...
        rasteramp(Ts(i), tmp_um(i)) + tmp_amp(i);
end

raster = zeros(totalT,length(C));
for j = 1:length(C)
    ind = find(I == C(j));
    inner_T = Ts(ind);
    for i = 1:length(inner_T)
        raster(inner_T(i),j) = raster(inner_T(i),j) + 1;
    end
end
% sort by max channel
[~, ind] = sort(cluster_um, 'ascend');
raster_sorted = raster(:,ind);


figure(1)
set(gcf, 'Position', [1 1 1200 580])
clf
subplot(1,3,1)
imagesc(raster1')
colormap('hot')
ylabel('10um bins')
subplot(1,3,2)
imagesc(raster_sorted')
colormap('hot')
ylabel('unit index')
xlabel('5 second bins')
subplot(1,3,3)
imagesc(sp')
ylabel('10um bins')



