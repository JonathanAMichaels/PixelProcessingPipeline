function wPCA = extractPCfromSnippets(rez, nPCs, nskip)
% extracts principal components for 1D snippets of spikes from all channels
% loads a subset of batches to find these snippets
% 
% [ks25] updates:
% - useMemMapping by default
% - corrected buffer usage in isolated_peaks_buffered.m  (prev. "isolated_peaks_new.m")
% - added override [nskip] input to subsample batches (sometimes more sparse/fine is desired)
% 
% ---
% 2021-xx-xx  TBC  Evolved from original Kilosort
% 2021-04-28  TBC  Cleaned & commented; useMemMapping = 1 by default
% 

ops = rez.ops;
useMemMapping = getOr(ops, 'useMemMapping',1);

if nargin<3 || isempty(nskip)
    % skip every this many batches
    nskip = getOr(ops, 'nskip', 25);
end

Nbatch      = ops.Nbatch;

% extract the PCA projections
CC = zeros(ops.nt0); % initialize the covariance of single-channel spike waveforms

if useMemMapping
    % use handle to memmapped file object
    fid = ops.fprocmmf;
else
    fid = fopen(ops.fproc, 'r'); % open the preprocessed data file
end

for ibatch = 1:nskip:Nbatch % from every nth batch
    
    dataRAW = get_batch(ops, ibatch, fid);

    % find isolated spikes from each batch
    [row, col] = isolated_peaks_buffered(dataRAW, ops);

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);

    c = sq(clips(:, :));
    CC = CC + gather(c * c')/1e3; % scale covariance down by 1,000 to maintain a good dynamic range
end

if ~useMemMapping
    fclose(fid); % clean up
end

[U] = svdecon(CC); % the singular vectors of the covariance matrix are the PCs of the waveforms

wPCA = U(:, 1:nPCs); % take as many as needed

% adjust the arbitrary sign of the first PC so its negativity is downward
% - this is strange...why manipulate sign of PC1, but not the others? --TBC
if sign(wPCA(ops.nt0min+1,1))>0
    fprintf(2, '~!~\tNotice:  sign flip of PC components detected & inverted during extractTemplatesfromSnippets.m operation...\n')
    % keyboard; % pause if unexpected polarity
    % ...turns out, this never seems to trigger. --TBC 2021
    % If it does, consider applying more complete sign change (e.g. /clustering/template_learning.m)
end
% wPCA(:,1) = - wPCA(:,1) * sign(wPCA(ops.nt0min+1,1));
wsign = -sign(wPCA(ops.nt0min, 1));
wPCA = wPCA .* wsign;

