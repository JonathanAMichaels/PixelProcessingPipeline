function rez = runTemplates(rez)

% this function will run Kilosort2 initializing at some previously found
% templates. These must be specified in rez.W, rez.U and rez.mu. The batch
% number at which the new template run is started is by default 1 (modify it by changing rez.istart). 

% If you don't mind changes in template throughout the recording, but want
% to use the results from a previous session after splitting and merging, just pass
% the rez output from this previous session.

% Keep in mind that these templates will nonetheless
% change substantially as they track neurons throughout the new recording. 
% As such, it is best to use the templates obtained at the very end of the
% previous recording, which are found in rez.WA(:,:,:,end),
% rez.UA(:,:,:,end), rez.muA(:,end). Keep in mind that any merging and
% splitting steps are done after running the templates, and would result in
% difference between sessions if done separately on each session. 

% update: these recommendations no longer apply for the datashift version!


if sum(isfield(rez, {'W', 'U', 'mu'}))<3
    error('missing at least one field: W, U, mu');
end

Nbatches = rez.ops.Nbatch;

% Tested "middle-out" method of extraction...no clear benefit
% % start extraction at targBatch, 
% % proceed in reverse-direction to start of file,
% % jump back to targBatch (recalling state at end of learning)
% % then sort in forward direction to end of file
if getOr(rez.ops, 'middleOut', 0)
    iorder = [(rez.ops.targBatch-1):-1:1, rez.ops.targBatch:1:Nbatches];
else
    iorder = 1:Nbatches;
end

rez.orderExtract = iorder;

[rez, st3, fW,fWpc] = trackAndSort(rez, iorder);

% sort all spikes by batch -- to keep similar batches together,
% which avoids false splits in splitAllClusters. Break ties 
% [~, isort] = sortrows(st3,[5,1,2,3,4]); 
% st3 = st3(isort, :);
% fW = fW(:, isort);
% fWpc = fWpc(:, :, isort);

% just display the total number of spikes
fprintf( 'Number of spikes extracted (before any postProcessing): %d\n', size(st3,1));

rez.st3 = st3;
rez.st2 = st3; % keep also an st2 copy, because st3 will be over-written by one of the post-processing steps

% the template features are stored in cProj, like in Kilosort1
rez.cProj    = fW';

%  permute the PC projections in the right order
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');
% iNeighPC keeps the indices of the channels corresponding to the PC features


%% Prep rez for postprocessing
% precompute any unique outputs of postprocessing stages so that they can be skipped as needed
% (...without breaking downstream analyses)!

Nfilt = size(rez.W,2); % new number of templates
Nrank = 3;
Nchan = rez.ops.Nchan;
NchanNear   = rez.ops.NchanNear;    %min(Nchan, 32);
Nnearest    = rez.ops.Nnearest;    %min(Nchan, 32);
nt0 = rez.ops.nt0;
nt0min = rez.ops.nt0min;
sigmaMask   = rez.ops.sigmaMask;

Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest ...
    Nrank 0 0 Nchan NchanNear nt0min 0]); % make a new Params to pass on parameters to CUDA

% determine what channels each template lives on
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear); 

% find the peak abs channel for each template
[~, iW] = max(abs(rez.dWU(nt0min, :, :)), [], 2);
iW = squeeze(int32(iW));

% we need to re-estimate the spatial profiles
[Ka, Kb] = getKernels(rez.ops, 10, 1); % we get the time upsampling kernels again
[rez.W, rez.U, rez.mu] = mexSVDsmall2(Params, rez.dWU, rez.W, iC-1, iW-1, Ka, Kb); % we run SVD

[WtW, iList] = getMeWtW(single(rez.W), single(rez.U), Nnearest); % we re-compute similarity scores between templates
rez.iList = iList; % over-write the list of nearest templates

rez.iNeigh   = gather(iList(:, 1:Nfilt)); % get the new neighbor templates
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt))); % get the new neighbor channels

% Calculate simScore including temporal lag
% - cranky implementation due to differences in rez.W and wPCA dimensions
%  - weird/deep inconsistencies & hardcoding of how many PCs are preserved (3, 6, 2*3?<!?)
rez.simScore = calc_SimScore(rez);

rez.Wphy = cat(1, zeros(1+nt0min, Nfilt, Nrank), rez.W); % for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of the window



%% Retain compressed time-varying templates
% - revived from A.Bondy Kilosort 2[.5?] fork
% this whole next block is just done to compress the compressed templates
% we separately svd the time components of each template, and the spatial components
% this also requires a careful decompression function, available somewhere in the GUI code

nKeep = min([Nchan*3,Nbatches,20]); % how many PCs to keep
rez.W_a = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b = zeros(Nbatches, nKeep, Nfilt, 'single');
rez.U_a = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b = zeros(Nbatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    % do this for every template separately
    WA = reshape(rez.WA(:, j, :, :), [], Nbatches);
    WA = gpuArray(WA); % svd on the GPU was faster for this, but the Python randomized CPU version might be faster still
    [A, B, C] = svdecon(WA);
    % W_a times W_b results in a reconstruction of the time components
    rez.W_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b(:,:,j) = gather(C(:, 1:nKeep));
    
    UA = reshape(rez.UA(:, j, :, :), [], Nbatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    % U_a times U_b results in a reconstruction of the time components
    rez.U_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b(:,:,j) = gather(C(:, 1:nKeep));
end

fprintf('Finished compressing time-varying templates \n')

end % main function
