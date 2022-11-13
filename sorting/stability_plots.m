clear

load('/Users/jonathanamichaels/Library/CloudStorage/Dropbox/mFiles-Projects/GitHub/PixelProcessingPipeline/geometries/neuropixPhase3B1_kilosortChanMap')
T = readNPY('spike_times.npy');
I = readNPY('spike_clusters.npy');
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

f = fopen('proc2.dat', 'r');
recordSize = 2; % 2 bytes for int16
nChan = 384;
spt = recordSize*nChan;

backSp = 20;
forwardSp = 60;
comChan = 8;
tmp_amp = zeros(1,length(T)); tmp_um = zeros(1,length(T));
for i = 1:length(T)
    if mod(i,100000) == 0
        disp(i)
    end
    fseek(f, (T(i)-backSp) * spt, 'bof');
    tmp = fread(f, [nChan, backSp+forwardSp], '*int16')';
    r = double(range(tmp,1));
    [m, ind] = sort(r, 'descend');
    norm_chan = m(1:comChan) / sum(m(1:comChan));
    tmp_um(i) = ceil((sum(ycoords(ind(1:comChan)) .* norm_chan')+1) / 10);
    tmp_amp(i) = mean(m(1:comChan));
end
fclose(f);

Ts = round(double(T)/30000)+1;
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


figure(2)
clf
hold on
cmap = lines(length(C));
for j = 1:length(C)
    disp(j)
    ind = find(I == C(j));
    theseSpikes = Ts(ind);
    ch = tmp_chan(ind);
    rn = randperm(length(theseSpikes));
    if length(rn) > 100
        rn = rn(1:100);
    end
    for i = 1:length(rn)
        plot(theseSpikes(rn(i)), ch(rn(i)), '.', 'Color', cmap(j,:), 'MarkerSize', 8)
    end
    axis([1 max(Ts) 0.5 384.5])
    drawnow
end


