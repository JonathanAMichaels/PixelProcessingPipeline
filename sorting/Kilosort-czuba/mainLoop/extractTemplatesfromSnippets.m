function [wTEMP, wPCA] = extractTemplatesfromSnippets(rez, nPCs, nskip)
% this function is very similar to extractPCfromSnippets.
% outputs not just the PC waveforms, but also the template "prototype", 
% basically k-means clustering of 1D waveforms.
% 
% [ks25] updates:
% - useMemMapping by default
% - corrected buffer usage in isolated_peaks_buffered.m  (prev. "isolated_peaks_new.m")
% - don't stop reading spike samples at arb. count, continue through all batches
% - added override [nskip] input to subsample batches (sometimes more sparse/fine is desired)
% 
% ---
% 2021-xx-xx  TBC  Evolved from original Kilosort
% 2021-04-28  TBC  Cleaned & commented; useMemMapping = 1 by default
% 

ops = rez.ops;
useMemMapping = getOr(ops, 'useMemMapping',1);
debugPlot =  getOr(ops, 'fig', 1)>=2;

if nargin<3 || isempty(nskip)
    % skip every this many batches
    nskip = getOr(ops, 'nskip', 25);
end

% more templates
nTEMP = getOr(ops, 'nTEMP', 2*nPCs);

Nbatch      = ops.Nbatch;

if useMemMapping
    % use handle to memmapped file object
    fid = ops.fprocmmf;
else
    fid = fopen(ops.fproc, 'r'); % open the preprocessed data file
end

k = 0;
dd = gpuArray.zeros(ops.nt0, 5e4, 'single'); % preallocate matrix to hold 1D spike snippets
for ibatch = 1:nskip:Nbatch
    % load batch of data directly to gpuArray
    dataRAW = get_batch(ops, ibatch, fid);

    % find isolated spikes from each batch
    [row, col] = isolated_peaks_buffered(dataRAW, ops);

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);

    c = sq(clips(:, :));

    if k+size(c,2)>size(dd,2)
        dd(:, 2*size(dd,2)) = 0;
    end

    dd(:, k + (1:size(c,2))) = c;
    k = k + size(c,2);

end

if ~useMemMapping
    fclose(fid); % clean up
end

% discard empty samples
dd = dd(:, 1:k);

% initialize the template clustering with sampling of waveforms **distributed throughout file duration**
% wTEMP = dd(:, randperm(size(dd,2), nPCs));
% wTEMP = dd(:, round(linspace(1, size(dd,2), nTEMP)));
sectSize = floor(size(dd,2)/nTEMP);
ti = randperm(sectSize, nTEMP) + round(linspace(0, size(dd,2)-sectSize, nTEMP));
wTEMP = dd(:, ti);
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % normalize them

if debugPlot
    % plot evolution of templates extracted from data
    figure(201);
    set(gcf, 'name','Kilosort [TEMP]lates', 'windowstyle','normal', 'position',[100,400, 2500, 200]);
    subplot(1,11,1), imagesc(wTEMP)
end

for i = 1:10
  % at each iteration, assign the waveform to its most correlated cluster
   cc = wTEMP' * dd;
   [amax, imax] = max(cc,[],1);
   for j = 1:nTEMP
      wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)'; % weighted average to get new cluster means
   end
   wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % unit normalize
   
   if i==10
       % sort templates by frequency on last iteration
       tc = histcounts(imax,nTEMP);
       [~,tord] = sort(tc,'descend');
       wTEMP = wTEMP(:,tord);
   end
   if debugPlot
       figure(201); subplot(1,11,i+1); imagesc(wTEMP);
   end
end



dd = double(gather(dd));
[U] = svdecon(dd); % the PCs are just the left singular vectors of the waveforms

wPCA = gpuArray(single(U(:, 1:nPCs))); % take as many as needed

% adjust the arbitrary sign of the first PC so its negativity is downward
% - this is strange...what is source/effect of this 
if sign(wPCA(ops.nt0min+1,1))>0
    fprintf(2, '~!~\tNotice:  sign flip of PC components detected & inverted during extractTemplatesfromSnippets.m operation...\n')
    %keyboard; % pause if unexpected polarity
    % ...turns out, this never seems to trigger. --TBC 2021
    % If it does, consider applying more complete sign change (e.g. /clustering/template_learning.m)
end
% wPCA(:,1) = - wPCA(:,1) * sign(wPCA(ops.nt0min+1,1));
wsign = -sign(wPCA(ops.nt0min, 1));
wPCA = wPCA .* wsign;

end %main function
