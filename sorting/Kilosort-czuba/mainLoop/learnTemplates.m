function rez = learnTemplates(rez, iorder)
% This is the main optimization. Takes the longest time and uses the GPU heavily.  

% changes (2022...mostly minor. --TBC)
% - added upperlimit on hardeningBatches

% Initialize various [missing] default ops
rez.ops.fig = getOr(rez.ops, 'fig', 1); % whether to show plots every N batches

% Turn on sorting of spikes before subtracting and averaging in mpnu8
rez.ops.useStableMode = getOr(rez.ops, 'useStableMode', 1);

% memory mapped file reads
rez.ops.useMemMapping = getOr(rez.ops, 'useMemMapping',0);

% min spike count before templates are updated 
rez.ops.clipMin     = getOr(rez.ops, 'clipMin', 0);
rez.ops.clipMinFit  = getOr(rez.ops,'clipMinFit',.7);


% lets try starting with a decent number of units
NrankPC = 6; % this one is the rank of the PCs, used to detect spikes with threshold crossings
Nrank = 3; % this one is the rank of the templates

rez.ops.LTseed = getOr(rez.ops, 'LTseed', 1);
rng('default'); rng(rez.ops.LTseed);


% DONE fiddling with rez.ops struct (this is sketchy)
% use shorthand from here on
ops = rez.ops;

%%

% create various local shorthand ops vars/flags
useStableMode       = ops.useStableMode;
useMemMapping       = ops.useMemMapping;
clipMin             = ops.clipMin;
clipMinFit          = ops.clipMinFit;


% we need PC waveforms, as well as template waveforms
cmdLog('Extracting initial templates from threshold crossings...', toc);
initSkip = 5; % be a little sparse in batch sampling for initial template creation
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC, initSkip);

% place PCA copies in rez struct
rez.wPCA = wPCA;
rez.wTEMP = wTEMP;

% move these to the GPU
wPCA = gpuArray(wPCA(:, 1:Nrank));
wTEMP = gpuArray(wTEMP);
wPCAd = double(wPCA); % convert to double for extra precision
% ops.wPCA = gather(wPCA);    % *** TBC: moved to rez; not sure why were being put in ops
% ops.wTEMP = gather(wTEMP);  % *** TBC: moved to rez; not sure why were being put in ops
nt0 = ops.nt0;
nt0min  = ops.nt0min;
nBatches  = ops.Nbatch;  %??? why rez.temp???  rez.temp.Nbatch;
NT  	= ops.NT;
Nfilt 	= ops.Nfilt;
Nchan 	= ops.Nchan;

batchstart = 0:NT:NT*nBatches;

% % min spike count before templates are updated 
% clipMin = ops.clipMin;

% number of batches to use as padding at end of learning phase
% - allows any newly added templates to settle and/or be trimmed before advancing to spike extraction
% - no new templates will be added during these batches
% - an equal number of additonal batches will be
% - upperlimit of 500 hardeningBatches
ops.hardeningBatches = getOr(ops, 'hardeningBatches', min(ceil(0.50 * nBatches), 500));
hardeningBatches = ops.hardeningBatches;

% decay of gaussian spatial mask centered on a channel
sigmaMask  = ops.sigmaMask;

% Prelim assessment of [physical] distance between channels
[~, ~, C2C] = getClosestChannels(rez, sigmaMask, min(ops.Nchan, 32));

% number of nearest channels to each primary channel
% [NchanNear] defines range of nearby channels that are considered when determining if threshold crossing is larger on another nearby channel
NchanNear   = 32;%min([2*ops.long_range(2), ops.Nchan, 32]); % [def: min(ops.Nchan, 32)]
% [Nnearest] defines number of channels templates span w/in cuda code
Nnearest    = 32;%find(mean(sort(C2C),2)<=500, 1, 'last'); % no reason for this to extend impossibly far, generous limit +/- 500 um
Nnearest    = min(Nnearest, 32); % cap at 32 chan (...support limit for GPU?)
% Both of these belong in [ops] struct
ops.NchanNear   = NchanNear;
ops.Nnearest    = Nnearest;

% spike threshold for finding missed spikes in residuals
% ops.spkTh = -6; % why am I overwriting this here?

% find the closest NchanNear channels, and the masks for those channels
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

% Construct sequence of batches [iorder] across all Learning Phases
iorder0 = iorder; % preserve the original iorder sequence
niter0 = length(iorder0);

batchPhase = ones(1,niter0);
% pad out end of learning phase with sequence of randomly selected batches to allow templates to be hardened
iorder = [iorder, randsample(iorder, hardeningBatches)];
batchPhase = [batchPhase, 2*ones(1, length(iorder)-length(batchPhase))];
% use equal duration of batch padding to walk templates back to starting batch of spike extraction
% - hardcoded to batch==1, but should allow for user override if alternate extractOrder is defined
% - (...or future method of using integer datashift epochs as sort

% Tested "middle-out" method of extraction...no clear benefit
p3min = min(hardeningBatches, 50); % nBatches in phase3
if getOr(rez.ops, 'middleOut', 0)
    if ops.targBatch <= (nBatches-p3min+1)
        iorder = [iorder, (ops.targBatch+p3min):-1:ops.targBatch];
    else
        % prevent error, but really shouldn't be used 
        iorder = [iorder, (ops.targBatch-p3min):1:ops.targBatch];
    end
else
    iorder = [iorder, p3min:-1:1];
end

batchPhase = [batchPhase, 3*ones(1, length(iorder)-length(batchPhase))];

% final count of total batches in learning phase
niter   = numel(iorder);

% Label which batches correspond to various learning phases
% - facilitates changes in processing based on current learning phase
% 1 = standard/initial learning phase 
% 2 = template hardening (no new templates, just allow any templates added near end of learning to evolve)
% 3 = conditioning templates for first batch of extraction (no new templates, no dropping)
% % % batchPhase = [ones(1,niter0), 2*ones(1,hardeningBatches), 3*ones(1, min(hardeningBatches,25))];
if length(batchPhase)~=niter, keyboard, end

% this is the absolute temporal offset in seconds corresponding to the start of the
% spike sorted time segment
t0 = ceil(ops.trange(1) * ops.fs);

nInnerIter  = 60; % this is for SVD for the power iteration

% schedule of learning rates for the model fitting part
% starts small and goes high, it corresponds approximately to the number of spikes
% from the past that were averaged to give rise to the current template
pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), hardeningBatches));
% - this parameter is presumably sensitive to the batch duration, in that long batches have greater potential to totally dominate waveform history
maxWeighting = 0.95; % never completely wipe out template history in one batch


Nsum = min(Nchan,7); % (def=7) how many channels to extend out the waveform in mexgetspikes

% lots of parameters passed into the CUDA scripts
% - AARRGGG@@!! Many of these inputs are just zombie values that are never used inside cuda functions they are passed into!?!@
%   - Unused: pmi
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ops.nt0min 0 Nsum NrankPC ops.Th(1) useStableMode]);

% W0 has to be ordered like this
W0 = permute(double(wPCA), [1 3 2]);

% initialize the list of channels each template lives on
iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(0,1, 'double');

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

% This
p1 = 0.8; %[def=0.95] ; % decay of nsp estimate in each batch

cmdLog('Learning templates...', toc);

if useMemMapping
    fid = ops.fprocmmf;
else
    fid = fopen(ops.fproc, 'r');
end

ndrop = zeros(1,3); % this keeps track of dropped templates for debugging purposes

m0 = ops.minFR * ops.NT/ops.fs; % this is the minimum firing rate that all templates must maintain, or be dropped

%% debug plot chan peaks
                            %
                            %
                            %

                            figure(99);
                            % position figure short & wide across bottom of screen
                            set(gcf, 'windowstyle','normal', 'position',[100,50, 2500, 300]);
                            chmax.block = 20000; % temporal samples to calc max across batch dat
                            chmax.batchsamp = floor((NT)/chmax.block);   %floor((NT+ops.ntbuff)/chmax.block);
                            chmax.hist = 50; % batches of history to retain in plot
                            chmax.tot = chmax.batchsamp*chmax.hist;
                            chmax.vals = nan(chmax.tot, ops.Nchan);
                            chmax.h = imagesc(chmax.vals');
                            set(chmax.h.Parent, 'xtick',1:chmax.batchsamp:chmax.tot,'xticklabel',vec2tick(1:chmax.hist, '%d '));
                            %
                            %
                            %
%%


%%
for ibatch = 1:niter        
    k = iorder(ibatch); % k is the index of the batch in absolute terms

    % obtained pm for this batch
    pm = pmi( min(ibatch,end));
    Params(9) = pm; % this doesn't seem to be used inside any cuda functions anymore.... (TBC 2021, ks25)

    
    % loading a single batch (same as everywhere)
    if useMemMapping % && isa(ops.fprocmmf, 'memmapfile')
        dataRAW = get_batch(ops, k, ops.fprocmmf);
    else
        dataRAW = get_batch(ops, k, fid);
    end
%    dataRAW = get_batch(ops, k, fid);
    Params(1) = size(dataRAW,1);
    
    
    if ibatch==1
       % only on the first batch, we first get a new set of spikes from the residuals,
       % which in this case is the unmodified data because we start with no templates       
       [dWU] = mexGetSpikes2_pcTight(Params, dataRAW, wTEMP, iC-1); % CUDA function to get spatiotemporal clips from spike detections
        dWU = double(dWU);
        dWU = reshape(wPCAd * (wPCAd' * dWU(:,:)), size(dWU)); % project these into the wPCA waveforms

        W = W0(:,ones(1,size(dWU,3)),:); % initialize the low-rank decomposition with standard waves
        Nfilt = size(W,2); % update the number of filters/templates
        nsp(1:Nfilt) = m0; % initialize the number of spikes for new templates with the minimum allowed value, so it doesn't get thrown back out right away
        Params(2) = Nfilt; % update in the CUDA parameters
        filterAge = ones(Nfilt,1); % weight updating based on batches since cluster creation
    end
    
    %% debug plot chan peaks
                                %
                                %
                                %
                                jnk = nan([chmax.batchsamp,ops.Nchan]);

                                tt = (0:chmax.block:NT)+1 + ops.ntbuff;
                                for i = 1:length(tt)-1
                                    jnk(i,:) = gather(max(dataRAW(tt(i):tt(i+1),:)));
                                end
                                chmax.vals = [chmax.vals(chmax.batchsamp+1:end,:); jnk];
                                chmax.h.CData = chmax.vals';
                                set(chmax.h.Parent, 'clim', prctile(chmax.vals(:),[5,99]));
                                set(chmax.h.Parent, 'xticklabel',vec2tick((-chmax.hist:0)+ibatch, '%d '));
                                drawnow nocallbacks
                                %
                                %
                                %
    %%
    
    % resort the order of the templates according to best peak channel
    % this is important in order to have cohesive memory requests from the GPU RAM
    [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2); % max channel (either positive or negative peak)
    iW = int32(squeeze(iW));
    
    [iW, isort] = sort(iW); % sort by max abs channel
    W = W(:,isort, :); % user ordering to resort all the other template variables
    dWU = dWU(:,:,isort);
    nsp = nsp(isort);
    filterAge = filterAge(isort);
    
    % !~!~!~!~!~!
    % THIS IS WHERE INVERSIONS OCCUR!!!
    % !~!~!~!~!~!
    % ...simple fix here won't work b/c size & order of prior W,U,mu changes as templates are learned
    
    % decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
    % this uses a "warm start" by remembering the W from the previous iteration
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);

    % %!~!~!~!~!~!
    % % NO) Don't want to reject based on shape, allow shape to do what it does
    % % - reject based on expected reliability of data (i.e. spike count) is much less heavy-handed
    % %     if ibatch>5
    % %         % detect inversion of 1st PC
    % %         dtest = dot(W(:,:,1), W01(:,:,1));
    % %         % If inversion detected, replace with prior W, U, mu
    % %         % - can't just invert W because it's been shifted by alignment applied w/in mexSVDsmall2 (cuda)
    % %         if any(dtest<=.5)
    % %             ii = find(dtest<=.5);
    % %             % figure(200)
    % %             %plot(dtest), title(mat2str(ii));
    % %             %drawnow
    % %             %keyboard
    % %             fprintf(2, '  batch %d\treverted PC inversion on unit %s\n',ibatch, mat2str(ii));
    % %             W(:,ii,:) = W01(:,ii,:);
    % %             U(:,ii,:) = U01(:,ii,:);
    % %             mu(ii) = mu01(ii);
    % %         end
    % %     end
    % %!~!~!~!~!~!
    
    
    % UtU is the gram matrix of the spatial components of the low-rank SVDs
    % it tells us which pairs of templates are likely to "interfere" with each other
    % such as when we subtract off a template
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); % this needs to change (but I don't know why!) (<---orig comment from Marius; was "why" or fix ever determined??)


    % main CUDA function in the whole codebase. does the iterative template matching
    % based on the current templates, gets features for these templates if requested (featW, featPC),
    % gets scores for the template fits to each spike (vexp), outputs the average of
    % waveforms assigned to each cluster (dWU0),
    % and probably a few more things I forget about
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp, errmsg] = ...
        mexMPnu8_pcTight(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA);
    
    
    % errmsg returns 1 if caller requested "stableMode" but mexMPnu8 was
    % compiled without the sorter enabled (i.e. STABLEMODE_ENABLE = false
    % in mexGPUAll). Send an error message to the console just once if this
    % is the case:
    if (ibatch == 1)
        if( (useStableMode == 1) && (errmsg == 1) )
            fprintf( 'useStableMode selected but STABLEMODE not enabled in compiled mexMPnu8.\n' );
        end
    end
    % Sometimes nsp can get transposed (think this has to do with it being
    % a single element in one iteration, to which elements are added
    % nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
    % to be a row vector.
    [nsprow, nspcol] = size(nsp);
    if nsprow<nspcol
        nsp = nsp';
    end


    % updates the templates as a running average weighted by recency
    %     % ----original----
    %     % since some clusters have different number of spikes, we need to apply the
    %     % exp(pm) factor several times, and fexp is the resulting update factor
    %     % for each template
    %     %     fexp = exp(double(nsp0).*log(pm));
    %     %     fexp = reshape(fexp, 1,1,[]);
    %     %     dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, []));
    %     % ----------------

    % % ----bad alt---- (overly restrictive on template shape)
    % %     % check for inversions in residual
    % %     [Wp] = mexSVDsmall2(Params, (dWU0./reshape(max(1, double(nsp0)), 1,1, [])), W, iC-1, iW-1, Ka, Kb);
    % %     % detect inversion of 1st PC
    % %     dtest = dot(W(:,:,1), Wp(:,:,1));
    % %     ii = find(gather(dtest<=.5) & gather(nsp0~=0)') ;
    % %     if ~isempty(ii)
    % %         fprintf(2, '  batch %d\tprevented PC inversion on unit %s\n',ibatch, mat2str(ii));
    % %     end
    % % ---------------
    
    % apply min spike count cutoff
    fexp = double(nsp0);        % fill w/counts from this batch
    
    isclip = (fexp>0) & (fexp<clipMin);
    ic1 = sum(isclip);
    for ggg = find(isclip)'
        g = id0==ggg; % spikes from this template
        % only continue with clip if spikes detected are not very good fits
        isclip(ggg) = median(vexp(g)./x0(g)) <= clipMinFit;
    end
    
    fexp(isclip) = 0;    % apply clipping    (leave nsp0 untouched so that clip does not affect template dropping)
    
    % updating weighting based on filter age
    pm = pmi(min(filterAge, end))';
    fexp = exp(fexp.*log(pm));  % exponentiate with weighting
    fexp = reshape(fexp.*maxWeighting, 1,1,[]) + (1-maxWeighting); % limit update weighting to [maxWeighting]% 
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, [])); % clipped updates will be canceled by zero weighting
    
    
    % nsp just gets updated according to the fixed factor p1
    if batchPhase(ibatch)==3
        % don't drop units as readily while returning to start
        p1 = 0.95;
    end
    nsp = nsp * p1 + (1-p1) * double(nsp0);

    
    
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if ops.fig>2
        figure(600)
        scatter(vexp./x0, double(nsp0(id0+1)), x0,'filled')
        set(gca, 'YScale','log','xlim',[.3,1])
        hold on, plot([0,1],clipMin*[1,1],'-','color',.85*[1 1 1]);
        hold off
        %     histogram(vexp./x0, .2:.02:1)
        title(sprintf('%d     [%d,  %d]', ibatch, ic1, sum(isclip)));
        drawnow
    end
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    

    
    if ibatch<niter
        % skip on very last pass, else error from repeat triageTemplates
        nextBatchPhase = batchPhase( min(ibatch+1,end));
        
        % Drop templates during initial learning(1) & hardening phases(2)
        if batchPhase(ibatch)<3 && (rem(ibatch, 5)==1 || nextBatchPhase==3)
            % this drops templates based on spike rates and/or similarities to other templates
            if nextBatchPhase==3
                % final triage opportunity
                % - this should cause any zero spike batches to be dropped, even if minFR==0
                nsp = nsp-eps;
            end
            [W, U, dWU, mu, nsp, ndrop, filterAge] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop, filterAge);
            if batchPhase(ibatch+1)==3
                % revert
                nsp = nsp+eps;
            end
            
        end
        Nfilt = size(W,2); % update the number of filters
        Params(2) = Nfilt;
        
        % DON'T add any new templates during hardening phases
        % - give new templates a chance to stabilize before moving on to extraction
        if batchPhase(ibatch)<2
            
            
            
            % adopt lower detection threshold for finding spikes in residuals
            pNew = Params;
            pNew(3) = mean(ops.Th);
            
            
            
            % this adds new templates if they are detected in the residual
            % [dWU0, dout] = mexGetSpikes2(Params, drez, wTEMP, iC-1);
            [dWU0] = mexGetSpikes2_pcTight(pNew, drez, wTEMP, iC-1);
            
            if size(dWU0,3)>0
                nNewFilt = size(dWU0,3);
                iNewFilt = Nfilt + [1:nNewFilt];
                
                ndrop(3) = .9 * ndrop(3) + .1 * nNewFilt; %nNewFilt; %
                
                % new templates need to be integrated into the same format as all templates
                dWU0 = double(dWU0);
                dWU0 = reshape(wPCAd * (wPCAd' * dWU0(:,:)), size(dWU0)); % apply PCA for smoothing purposes
                dWU = cat(3, dWU, dWU0);
                % initialize new templates with features from residuals that created them
                %      W(:, iNewFilt, :) = W0(:, ones(1,nNewFilt), :); % initialize temporal components of waveforms
                W(:, iNewFilt, :) = W0(:, ones(1,nNewFilt), :); % initialize temporal components of waveforms
                for nf = 1:length(iNewFilt)
                    t = iNewFilt(nf);
                    [w,s,u] = svdecon(dWU0(:,:,nf));
                    wsign = -sign(w(ops.nt0min, 1));
                    W(:,t,:) = wsign * w(:,1:Nrank);
                    U(:,t,:) = wsign * u(:,1:Nrank) * s(1:Nrank,1:Nrank);
                    mu(t)    = sum(sum(U(:,t,:).^2))^.5;
                    U(:,t,:) = U(:,t,:) / mu(t);
                end
                
                % % print mean amplitude of new templates
                % fprintf('\n   %s   ', mat2str(mu(iNewFilt)',3)); 
                
                nsp(iNewFilt) = ops.minFR * NT/ops.fs; % initialize the number of spikes with the minimum allowed
                mu(iNewFilt)  = 10; % initialize the amplitude of this spike with a lowish number
                filterAge(iNewFilt) = 0;
                
                Nfilt = min(ops.Nfilt, size(W,2)); % if the number of filters exceed the maximum allowed, clip it
                if Nfilt==ops.Nfilt
                    fprintf(2, '~!~\ttemplate count clipped at max allowed (%d, %d lim)\t[batch %d=%d]\n', size(W,2), Nfilt, k,ibatch);
                end
                Params(2) = Nfilt;
                
                % remove any new filters over the maximum allowed
                W   = W(:, 1:Nfilt, :); 
                U   = U(:, 1:Nfilt, :);
                dWU = dWU(:, :, 1:Nfilt);
                nsp = nsp(1:Nfilt);
                mu  = mu(1:Nfilt);
            end
        end
        if batchPhase(ibatch)<3        
            % increment filter ages
            filterAge = filterAge+1;
        end
    end
    
    if ops.fig>2
        figure(199)
        plot(sort(filterAge))
        ylim([0,niter]); xlim([0,ops.Nfilt])
        drawnow
    end
        
    % generously report status in command window
    doRefresh = (ibatch<10) ...
             || (ibatch<20 && rem(ibatch, 10)==1) ...
             || (ibatch<200 && rem(ibatch, 20)==1) ...
             || (rem(ibatch, 50)==1) ...
             || diff(batchPhase(ibatch+[-1,0]))~=0 ... last batch of current Learning Phase
             || ibatch==niter ... final batch
             ;
    if doRefresh    %rem(ibatch, 100)==1
        % this is some of the relevant diagnostic information to be printed during training
        thisStr = sprintf('%3d / %d batches, phase %d, %d units,\t nspks: %7.2f, mu: %6.4f, nst0: %4d, merges: %2.3f, %2.3f, %2.3f', ...
            ibatch, niter, batchPhase(ibatch), Nfilt, sum(nsp), median(mu), numel(st0), ndrop);
        cmdLog(thisStr, toc);

        % Update diagnostic figures
        if ops.fig
            if ibatch==1 || diff(batchPhase(ibatch+[-1,0]))~=0
                if exist('figHand','var') && getOr(ops, 'fig', 1) % && evalin('base','exist(''figDir'',''var'');')
                    try
                        % save existing figure
                        [~,fn] = fileparts(ops.saveDir);
                        % - No PDF save ....for some reason, kilosort gui fig keeps taking over figure focus during PDF save
                        figureFS({figHand, [get(figHand,'name'),'-',fn]});
                        set(figHand, 'tag', fullfile(ops.saveDir,'figs')); % embed default save destination (used by saveFigTriplet.m)
                        saveFigTriplet(0, [], {'mat','png'});
                    end
                end
                % new figure at start and new Learning Phase
                figHand = figure;
                set(figHand,'name',sprintf('learnTemplates (P%d)',batchPhase(ibatch)))
                addFigInfo(ops, figHand);
            else
                figure(figHand);
            end
            make_fig(W, U, mu, nsp, ibatch)
        end
    end
end

if ~useMemMapping
    fclose(fid);
end
toc

%%
if ops.fig>2
    figure(199)
    hold on
    plot(filterAge,'.-')
    xlim([0,length(filterAge)+1])
    hold off
end

% final covariance matrix between all templates
[WtW, iList] = getMeWtW(single(W), single(U), Nnearest);
% iW is the final channel assigned to each template
[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));
% the similarity score between templates is simply the correlation,
% taken as the max over several consecutive time delays
rez.simScore = gather(max(WtW, [], 3));

rez.iNeighPC    = gather(iC(:, iW));

% the neihboring templates indices are stored in iNeigh
rez.iNeigh   = gather(iList);


% Memorize the state of the templates at this timepoint.
rez = memorizeW(rez, W, dWU, U, mu); % memorize the state of the templates
rez.ops = ops; % update these (only rez comes out of this script)
rez.nsp = nsp;
rez.orderLearned = iorder;

% save('rez_mid.mat', 'rez');

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


cmdLog('Finished learning templates.')

