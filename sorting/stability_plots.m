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


f = fopen('proc.dat', 'r');
recordSize = 2; % 2 bytes for int16
nChan = 384;
spt = recordSize*nChan;

% gather SD per channel
times = randsample(max(T),10000,false);
samples = zeros(length(times),384);
for i = 1:length(times)
    fseek(f, times(i) * spt, 'bof');
    samples(i,:) = fread(f, [nChan, 1], '*int16');  
end
sd = std(double(samples),[],1);

comChan = 16;
tmp_amp = zeros(1,length(T)); tmp_um = zeros(1,length(T));
for i = 1:length(T)
    if mod(i,100000) == 0
        disp(i)
    end
    fseek(f, T(i) * spt, 'bof');
    tmp = abs(double(fread(f, [nChan, 1], '*int16')))';
    tmp = tmp ./ sd;
    tmp(tmp < 5) = 0;
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

figure(1)
clf
subplot(2,1,1)
imagesc(raster1')
colormap('hot')
subplot(2,1,2)
imagesc(rasteramp')
colormap('hot')

