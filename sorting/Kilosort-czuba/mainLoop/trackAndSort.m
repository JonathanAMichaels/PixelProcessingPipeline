function [rez, st3, fW, fWpc] = trackAndSort(rez, iorder)
% This is the extraction phase of the optimization. 
% iorder is the order in which to traverse the batches

% Turn on sorting of spikes before subtracting and averaging in mpnu8
rez.ops.useStableMode = getOr(rez.ops, 'useStableMode', 1);

ops = rez.ops;

% create various local shorthand ops vars/flags
useStableMode = ops.useStableMode;
useMemMapping = getOr(ops, 'useMemMapping',1);

% revert to the saved templates
W = gpuArray(rez.W);
U = gpuArray(rez.U);
mu = gpuArray(rez.mu);

Nfilt 	= size(W,2);
nt0 = ops.nt0;
Nchan 	= ops.Nchan;

dWU = gpuArray.zeros(nt0, Nchan, Nfilt, 'double');
for j = 1:Nfilt
    dWU(:,:,j) = mu(j) * squeeze(W(:, j, :)) * squeeze(U(:, j, :))';
end

% preserve state of learned templates
rezLrn = struct();
rezLrn = memorizeW(rezLrn, W, dWU, U, mu);
% ...add this to rez struct at end of this fxn

ops.fig = getOr(ops, 'fig', 1); % whether to show plots every N batches

NrankPC = 6; % this one is the rank of the PCs, used to detect spikes with threshold crossings
Nrank   = 3; % this one is the rank of the templates
rng('default'); rng(1); % initializing random number generator

% move these to the GPU
wPCA = gpuArray(rez.wPCA);

nt0min  = rez.ops.nt0min;
rez.ops = ops;
nBatches  = ops.Nbatch;
NT  	= ops.NT;

% min spike count before templates are updated 
clipMin     = getOr(ops, 'clipMin', 0);
clipMinFit  = getOr(ops,'clipMinFit',.7);

% Number of nearest channels to each primary channel
NchanNear   = getOr(ops, 'NchanNear', min(ops.Nchan, 32));
Nnearest    = getOr(ops, 'Nnearest', min(ops.Nchan, 32));

% decay of gaussian spatial mask centered on a channel
sigmaMask  = ops.sigmaMask;

% spike threshold for finding missed spikes in residuals
% % % ops.spkTh = -12; %-6 % why am I overwriting this here?
% % % % crank up this spkTh limit so we don't inject garbage into templates

batchstart = 0:NT:NT*nBatches;

% find the closest NchanNear channels, and the masks for those channels
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

niter   = numel(iorder);

% this is the absolute temporal offset in seconds corresponding to the start of the
% spike sorted time segment
% - also account for ntbuff (which is now loaded/discarded appropriately)
t0 = ceil(rez.ops.trange(1) * ops.fs);% - ops.ntbuff;

nInnerIter  = 60; % this is for SVD for the power iteration

% schedule of learning rates for the model fitting part
% starts small and goes high, it corresponds approximately to the number of spikes
% from the past that were averaged to give rise to the current template
% !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
% !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
% Make template dynamics atleast as 'hard' as they were at end of learning
pm = exp(-1/ (ops.momentum(2)) );
maxWeighting = 0.8; % never completely wipe out template history in one batch

% !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
% !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!

Nsum = min(Nchan,7); % (def=7) how many channels to extend out the waveform in mexgetspikes

% lots of parameters passed into the CUDA scripts
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pm Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1) useStableMode]);

% extract ALL features on the last pass
Params(13) = 2; % this is a flag to output features (PC and template features)

% different threshold on last pass?
Params(3) = ops.Th(end); % usually the threshold is much lower on the last pass

% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(Nfilt,1, 'double');

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate in each batch

% the list of channels each template lives on
% also, covariance matrix between templates
[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));
[WtW, iList] = getMeWtW(single(W), single(U), Nnearest);

cmdLog('Applying learned templates ...', toc);

fid = fopen(ops.fproc, 'r');

% allocate variables for collecting results
st3 = zeros(1e7, 5); % this holds spike times, clusters and other info per spike
ntot = 0;

% these next three store the low-d template decompositions
rez.WA = zeros(nt0, Nfilt, Nrank,nBatches,  'single');
rez.UA = zeros(Nchan, Nfilt, Nrank,nBatches,  'single');
rez.muA = zeros(Nfilt, nBatches,  'single');
% rolling count of new spikes per template per batch
rez.nspA = zeros(Nfilt, nBatches,  'single');

% plot for template inversion detection
if ops.fig>1
    Hsp = mkTemplateDebugFig;
end
invDetected = false(Nfilt, nBatches);



% these ones store features per spike
fW  = zeros(Nnearest, 1e7, 'single'); % Nnearest is the number of nearest templates to store features for
fWpc = zeros(NchanNear, Nrank, 1e7, 'single'); % NchanNear is the number of nearest channels to take PC features from


dWU1 = dWU;
troubleUnits = [];

for ibatch = 1:niter    
    k = iorder(ibatch); % k is the index of the batch in absolute terms
    
    % Tested "middle-out" method of extraction...no clear benefit
    if getOr(rez.ops, 'middleOut', 0) && k==rez.orderLearned(end)
        % revert to state at end of learning
        fprintf(2, '\n\t~~!!~~\tReverting to learned state for batch %d(#%d) ~~!!~~\n', ibatch, k);
        W = gpuArray(rezLrn.W);
        U = gpuArray(rezLrn.U);
        mu = gpuArray(rezLrn.mu);
        dWU = rezLrn.dWU;
    end
        
        
    % loading a single batch (same as everywhere)
    %     offset = 2 * ops.Nchan*batchstart(k);
    %     fseek(fid, offset, 'bof');
    %     dat = fread(fid, [ops.Nchan NT + ops.ntbuff], '*int16')';
    %     %dat = dat';
    %     dataRAW = single(gpuArray(dat))/ ops.scaleproc;
    if useMemMapping % && isa(ops.fprocmmf, 'memmapfile')
        dataRAW = get_batch(ops, k, ops.fprocmmf);
    else
        dataRAW = get_batch(ops, k, fid);
    end    
    %     dataRAW = get_batch(ops, k, fid);
    
    Params(1) = size(dataRAW,1);
    
    % !~!~!~!~!~!
    % THIS IS WHERE INVERSIONS OCCUR!!!
    % !~!~!~!~!~!
    % Preserve previous svd
    %     if ibatch==1
    %         W01 = rez.W;
    %         U01 = U;
    %         mu01 = mu;
    %     else
        W01 = W;
        U01 = U;
        mu01 = mu;        
        %     end

    % decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
    % this uses a "warm start" by remembering the W from the previous
    % iteration
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);
    
    % detect inversion of 1st PC
    dtest = dot(W(:,:,1), W01(:,:,1));
    % If inversion detected, replace with prior W, U, mu
    % - can't just invert W because it's been shifted by alignment applied w/in mexSVDsmall2 (cuda)
    if any(dtest<=.5)
        ii = find(dtest<=.5);
        % keep record of units with inversion problems & what batch they occurred on
        % - these get passed to [rez] output struct at end of this function
        % - plot them afterward with:  plotTemplateDynamics( rez, rez.troubleUnits);
        troubleUnits = unique([troubleUnits, ii]);
        invDetected(ii,k) = true;
        if ops.fig>1 % debug plotting
            try
                plot(Hsp, dtest);   hold on
                text(Hsp, 1,0.3, sprintf('batch  %3d\ninversions:  %s', k, mat2str(ii)), 'fontsize',10);
                set(Hsp, 'ylim',[-.1,1.1]);
                hold off,   drawnow nocallbacks
            catch
                Hsp = mkTemplateDebugFig;
            end
        end
        
        %         fprintf(2, '  batch %d\treverted PC inversion on unit %s\n',ibatch, mat2str(ii));
        
        
        % !~!~!~!~!~!
        % this SVD reversion doesn't address the *cause*, but DOES FIX the resulting batch stuttering!!!
        % !~!~!~!~!~!
        W(:,ii,:) = W01(:,ii,:);
        U(:,ii,:) = U01(:,ii,:);
        mu(ii) = mu01(ii);
    end
    
    % UtU is the gram matrix of the spatial components of the low-rank SVDs
    % it tells us which pairs of templates are likely to "interfere" with each other
    % such as when we subtract off a template
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); % this needs to change (but I don't know why!)
    
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % main CUDA function in the whole codebase. does the iterative template matching
    % based on the current templates, gets features for these templates if requested (featW, featPC),
    % gets scores for the template fits to each spike (vexp), outputs the average of
    % waveforms assigned to each cluster (dWU0),
    % and probably a few more things I forget about
    
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp, errmsg] = ...
        mexMPnu8_pcTight(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA);
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % Sometimes nsp can get transposed (think this has to do with it being
    % a single element in one iteration, to which elements are added
    % nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
    % to be a row vector.
    [nsprow, nspcol] = size(nsp);
    if nsprow<nspcol
        nsp = nsp';
    end
    

    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % updates the templates as a running average weighted by recency
    % since some clusters have different number of spikes, we need to apply the
    % exp(pm) factor several times, and fexp is the resulting update factor
    % for each template
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % Change from kilosort 2.0 to 2.5: Templates no longer updated/refined during extraction
    % - Instead of weighted update of [dWU] that occurs during learning phase,
    %   [dWU0] output from mexMPnu8 just gets dumped into [dWU1] instead
    % - ks25: Revive temporally dynamic template updates extraction
    %   - update using single/final 'momentum' value [pm] (set *once* prior to batch loop)
    
    % apply min spike count cutoff
    fexp = double(nsp0);        % fill w/counts from this batch
    
    % if too few spikes with not great fits were detected on a particular template, don't use them for update 
    isclip = (fexp>0) & (fexp<clipMin);
    ic1 = sum(isclip);
    for ggg = find(isclip)'
        g = id0==ggg; % spikes from this unit
        % only continue with clip if spikes detected are not very good fits
        isclip(ggg) = median(vexp(g)./x0(g)) <= clipMinFit;
    end
    fexp(isclip) = 0;    % apply clipping to updating weights (...leaving nsp0 intact)
    
    fexp = exp(fexp.*log(pm));  % exponentiate with weighting
    fexp = reshape(fexp.*maxWeighting, 1,1,[]) + (1-maxWeighting); % limit update weighting to [maxWeighting]% 
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, [])); % clipped updates will be canceled by zero weighting
    
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % still allow accumulated dWU1
    % - BUT instead of injecting it as [rez.dWU] at end, just dump it into separate [rez.dWU1] for posterity
    dWU1 = dWU1  + dWU0;
    
    nsp = nsp + double(nsp0);
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
    % !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
        
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    % during the final extraction pass, this keep track of template temporal dynamics & spikes per batch
    
    % we memorize the spatio-temporal decomposition of the waveforms at this batch
    % this is currently only used in the GUI to provide an accurate reconstruction
    % of the raw data at this time
    rez.WA(:,:,:,k) = gather(W);
    rez.UA(:,:,:,k) = gather(U);
    rez.muA(:,k)    = gather(mu);
    rez.nspA(:,k)   = gather(nsp0);

    % we carefully assign the correct absolute times to spikes found in this batch
    toff = nt0min + t0 + NT*(k-1);
    st = toff + double(st0)  - ops.ntbuff;

    % Trim output to batch spikes, excluding any spikes w/in the pre/post buffer samples (ops.ntbuff)
    bs = (st<toff) | (st>=toff+NT); % index of buffer spikes to be excluded
    % apply to new spike outputs
    st(bs)  = [];
    id0(bs) = [];
    x0(bs)  = [];
    vexp(bs)= [];
    
    irange = ntot + [1:numel(x0)]; % spikes and features go into these indices
    
    if ntot+numel(x0)>size(st3,1)
        % if we exceed the original allocated memory, double the allocated sizes
        fW(:, 2*size(st3,1))    = 0;
        fWpc(:,:,2*size(st3,1)) = 0;
        st3(2*size(st3,1), 1)   = 0;
    end 
    
    st3(irange,1) = double(st);     % spike times
    st3(irange,2) = double(id0+1);  % spike clusters (1-indexing)
    st3(irange,3) = double(x0);     % template amplitudes
    st3(irange,4) = double(vexp);   % residual variance of this spike
    st3(irange,5) = k;         % batch from which this spike was found
    
    fW(:, irange) = gather(featW(:,~bs));           % template features for this batch
    fWpc(:, :, irange) = gather(featPC(:,:,~bs));   % PC features
    
    ntot = ntot + numel(x0); % keeps track of total number of spikes so far
    
    
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if ops.fig>2
        figure(600)
        %histogram(vexp./x0, .2:.02:1)
        scatter(vexp./x0, double(nsp0(id0+1)), x0,'filled')
        set(gca, 'YScale','log','xlim',[.3,1])
        hold on, plot([0,1],clipMin*[1,1],'-','color',.85*[1 1 1]);
        hold off
        title(sprintf('%d     [%d,  %d]', ibatch, ic1, sum(isclip)));
        drawnow
    end
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    
    % generously report status in command window
    doRefresh = (ibatch<10) ...
             || (ibatch<100 && rem(ibatch, 10)==1) ...
             || (ibatch<200 && rem(ibatch, 20)==1) ...
             || (rem(ibatch, 50)==1) ...
             || ibatch==niter;
    if doRefresh    %rem(ibatch, 100)==1
        % this is some of the relevant diagnostic information to be printed during training
        %         fprintf('%2.2f sec, %d / %d batches, %d units, nspks: %d, mu: %2.4f, nst0: %d \n', ...
        %             toc, ibatch, niter, Nfilt, ntot, median(mu), numel(st0))
        thisStr = sprintf('%3d / %d batches,\t %d units, nspks: %7.2f, mu: %6.4f, nst0: %4d', ...
            ibatch, niter, Nfilt, ntot, median(mu), numel(st0));
        cmdLog(thisStr, toc);
        
        % these diagnostic figures should be mostly self-explanatory
        if ops.fig
            if ibatch==1 || ~exist('figHand','var')
                figHand = figure;
                set(figHand,'name','extractTemplates')
                addFigInfo(ops, figHand);
            else
                figure(figHand);
            end            
            
            make_fig(W, U, mu, nsp, ibatch)
        end
    end
end
fclose(fid);

% discards the unused portion of the arrays
st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

% !~!~!~!~!~!~!~!~!~!~!~!~!!~!~!~!~!~!~!~!
% ks25 fix:
% output proper dWU to rez
% - note: dWU is already normalized by count, so no need for same scaling operation as [dWU1]
rez.dWU = dWU;

rez.dWU1 = dWU1 ./ reshape(nsp, [1,1,Nfilt]);
rez.nsp = nsp;

% Output template state prior to extraction (i.e. as learned, prior to any temporal dynamics during extraction)
% Ensure all GPU arrays are transferred to CPU side
rl_fields = fieldnames(rezLrn);
for i = 1:numel(rl_fields)
    field_name = rl_fields{i};
    if(isa(rezLrn.(field_name), 'gpuArray'))
        rezLrn.(field_name) = gather(rezLrn.(field_name));
    end
end
rez.rezLrn = rezLrn;
rez.troubleUnits = troubleUnits;
rez.invDetected = invDetected;

% save figure to saveDir
if exist('figHand','var') && getOr(ops, 'fig', 1) % && evalin('base','exist(''figDir'',''var'');')
    try
        % save existing figure
        % - No PDF save ....for some reason, kilosort gui fig keeps taking over figure focus during PDF save
        [~,fn] = fileparts(ops.saveDir);
        figureFS({figHand, [get(figHand,'name'),'-',fn]});
        set(figHand, 'tag', fullfile(ops.saveDir,'figs')); % embed default save destination (used by saveFigTriplet.m)
        saveFigTriplet(0, [], {'mat','png'});
    end
end

% Done.
cmdLog( sprintf('\t%s complete', mfilename));


% Nested function for debug plot creation
function Hsp = mkTemplateDebugFig
figure(201);
set(gcf, 'windowstyle','normal', 'position',[100,400, 2500, 200]);
Hsp = subplot(1,11,[3:9]); % plot into axes handle
axis tight
set(Hsp, 'ylim',[-.1,1.1], 'ticklength',[.01 .005]);
end

end % main function
