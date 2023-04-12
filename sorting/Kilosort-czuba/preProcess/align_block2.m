function [imin,yblk, F0, F0m, targBatch] = align_block2(F, ysamp, ops)
% 
% F is y bins by amp bins by batches
% ysamp are the coordinates of the y bins in um
% 
% - updated to take ops struct as third input (instead of just receiving ops.nblocks)
% - [ops.targBatch] = user selected starting batch for 'datashift' alignment
%                     (defaults to batch 2/3 into file)
% - add [targBatch] output
% 

nblocks = max(ops.nblocks,1);
% nblocks must be >= 1 for proper initialization, but allow nblocks==0 in cases where
% we want to compute drift estimates, but not [necessarily] apply them
debugPlot = ops.fig>1;

Nbatches = size(F,3);
ysp = median(diff(ysamp));

% look up and down this many y bins to find best alignment
n = 25; % [15] % this val only alters alignment frame selection
dc = zeros(2*n+1, Nbatches);
dt = -n:n;

% we do everything on the GPU for speed, but it's probably fast enough on
% the CPU
Fg = gpuArray(single(F));

% mean subtraction to compute covariance
Fg = Fg - mean(Fg, 1);

% initialize the target "frame" from a single batch
% - allow to define relative location of target batch in file (...allowing flexibility for batch size)
% - default midway into file
targBatch = getOr(ops, 'targBatch', 0.5)
if targBatch<1
    % set to nearest batch
    targBatch = floor(size(Fg,3) * targBatch);
else
    % just enforce limit for error correction
    targBatch = min(round(targBatch), size(Fg,3));
end

% smooth across neighboring batches before proceeding
j = ceil(.05 * Nbatches);
F0 = my_conv2(Fg(:,:, targBatch+[-j:j]), 1, 3);
F0 = F0(:,:,j+1);

% first we do rigid registration by integer shifts
% everything is iteratively aligned until most of the shifts become 0. 
niter = 10;

% project initial onto this batch, then plot as baseline
dc0 = zeros(2*n+1, Nbatches);
for t = 1:length(dt)
    % for each NEW potential shift, estimate covariance
    dc0(t, :) = gather(sq(mean(mean(circshift(Fg, dt(t), 1) .* F0, 1), 2)));
end

if ops.fig
    H = figure;
    spx = 11; spy = niter+1;
    subplot(spy, spx, 1:spx-1)
    imagesc(1:Nbatches, dt*ysp, dc0);
    set(gca, 'YDir','normal');
    ylabel('baseline cov')
    hold on
    plot(targBatch,0,'k*');
    box off
    subplot(spy, spx, spx)
    imagesc(F0);
    box off
end

dall = zeros(niter, Nbatches);
% sort hack to bias drift estimate toward zero when multiple max values exist
% - particularly when all are zero in cases of zeroed out artifacts
[~, dtOrdi] = sort(abs(dt));
dtOrd = dt(dtOrdi);
for iter = 1:niter    
    for t = 1:length(dt)
        % for each NEW potential shift, estimate covariance
        Fs = circshift(Fg, dt(t), 1);
        dc(t, :) = gather(sq(mean(mean(Fs .* F0, 1), 2)));
    end
    
    if iter<niter
        % apply decreasing amount of smoothing across iterations
        % - last two iter effectively unsmoothed; (<0.25 clipped inside my_conv2.m)
        dcs = my_conv2(dc, .1*(niter-iter));

        % up until the very last iteration, estimate the best shifts
        [~, imax] = max(dcs(dtOrdi,:), [], 1);
        imax = dtOrdi(imax); % convert sorted index to original index
        
        % align the data by these integer shifts
        for t = 1:length(dt)
            ib = imax==t;
            Fg(:, :, ib) = circshift(Fg(:, :, ib), dt(t), 1);
            dall(iter, ib) = dt(t);
        end
        
        % % % % new target frame based on our current best alignment
        % % %  F0 = mean(Fg, 3);

        % Scale breadth of new target frame across iterations
        ii = targBatch + ceil([-.5 .5]*Nbatches/niter*(iter^2));
        ii(ii<1) = 1;   ii(ii>Nbatches) = Nbatches; % constrain valid indices
        F0 = mean(Fg(:,:, ii(1):ii(2)), 3);

    end
    if ops.fig
        figure(H);
        subplot(spy, spx, (1:spx-1)+iter*spx);     % subplot(niter+1, 1, iter+1)
        imagesc(1:Nbatches, dt*ysp, dc);
        set(gca, 'YDir','normal');
        hold on
        % plot shift from this iteration
        plot(1:Nbatches, dall(iter,:).*ysp, 'r-', 'linewidth',1);
        % identify target batch region
        plot(ii, [.8 .8]*max(ylim), '--w', 'linewidth',1);
        box off
        subplot(spy, spx, iter*spx+spx)
        imagesc(F0);
        drawnow
    end

end


% new target frame based on our current best alignment
F0 = mean(Fg, 3);

% now we figure out how to split the probe into nblocks pieces
% if nblocks = 1, then we're doing rigid registration
nybins = size(F,1);
yl = floor(nybins/nblocks)-1;
ifirst = round(linspace(1, nybins - yl, 2*nblocks-1));
ilast  = ifirst + yl; %287;

%% smoothing across blocks

%nblocks = length(ifirst);
yblk = zeros(length(ifirst), 1);

% this is really only informative with multiple blocks (...maybe)
if ops.fig && nblocks>1
    H = figure;
    subplot(5,1,1:3)
    imagesc(1:Nbatches, dt*ysp, dc);
    set(gca, 'YDir','normal');
    hold on
    plot(targBatch,0,'r*');
    box off
end

% for each small block, we only look up and down this many samples to find
% nonrigid shift
n = 5;
dt = -n:n;

% this part determines the up/down covariance for each block without
% shifting anything
dcs = zeros(2*n+1, Nbatches, nblocks);
for j = 1:nblocks
    isub = ifirst(j):ilast(j);
    yblk(j) = mean(ysamp(isub));
    
    Fsub = Fg(isub, :, :);
     
    for t = 1:length(dt)
        Fs = circshift(Fsub, dt(t), 1);
        dcs(t, :, j) = gather(sq(mean(mean(Fs .* F0(isub, :, :), 1), 2)));
    end
end
% plot initial
if ops.fig && nblocks>1
    % flatten dcs for image plot
    dcsFlat = squeeze(num2cell(dcs,[1,2]));
    dcsFlat = cat(1, dcsFlat{:});
    figure(H);
    subplot(5,1,4)
    imagesc(1:Nbatches, dt*ysp*Nbatches, dcsFlat);
    set(gca, 'YDir','normal');
    ylabel('initial')
    box off
end

% to find sub-integer shifts for each block , 
% we now use upsampling, based on kriging interpolation
dtup = linspace(-n, n, (2*n*10)+1);    
K = kernelD(dt,dtup,1); % this kernel is fixed as a variance of 1
dcs = my_conv2(dcs, .5, [1, 2, 3]); % some additional smoothing for robustness, across all dimensions
% plot smoothed
if ops.fig && nblocks>1
    % flatten dcs for image plot
    dcsFlat = squeeze(num2cell(dcs,[1,2]));
    dcsFlat = cat(1, dcsFlat{:});
    figure(H);
    subplot(5,1,5)
    imagesc(1:Nbatches, dt*ysp, dcsFlat);
    set(gca, 'YDir','normal');
    ylabel('smoothed')
    box off
end

imin = zeros(Nbatches, nblocks);
for j = 1:nblocks
    % using the upsampling kernel K, get the upsampled cross-correlation
    % curves
    dcup = K' * dcs(:,:,j);
    
    % find the  max of these curves
    [~, imax] = max(dcup, [], 1);
    
    % add the value of the shift to the last row of the matrix of shifts
    % (as if it was the last iteration of the main rigid loop )
    dall(niter, :) = dtup(imax);
    
    % the sum of all the shifts equals the final shifts for this block
    imin(:,j) = sum(dall,1);
end


%%
Fg = gpuArray(single(F));
imax = sq(sum(dall(1:niter-1,:),1));
for t = 1:length(dt)
    ib = imax==dt(t);
    Fg(:, :, ib) = circshift(Fg(:, :, ib), dt(t), 1);
end
F0m = mean(Fg,3);


