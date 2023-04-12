function rez = datashift2(rez, do_correction)
% function rez = datashift2(rez, do_correction)
% 
% ~~~[ missing description in original Kilosort 2.5 source ]~~~
% 
% 
%
% [ks25] updates:           (...oh man, so many...)
% MAJOR:
% - [do_correction] input flag a little hackish, but behavior mostly consistent w/orig usage
%   - >=0   will always compute drift correction
%   - ==1   will apply drift correction directly to preprocessed data (rez.ops.fproc), as normal
%   - <0    will skip completely (former ==0), but not recommended
%           (better to KNOW shift status, even if not applied)
% - [ops.integerShifts] flag to constrain any shifts to integer multiples of y-axis electrode spacing (def=0)
% - 
% - revised spike threshold binning from original fixed range logspaced(10:100) to
%   optimized log spaced range spanning 99% ci of all contributing spike amplitudes
%   - range automatically scaled up to ensure unique bin centers
%   - esp important b/c spike amp outputs from standalone_detector.m are **integers**
% - fixed error in standalone_detector.m >> spikedetector3.cu that silently ignored large batches of data(!!!)
%   - see https://github.com/MouseLand/Kilosort/issues/394
% >> (align_block2.m)
% - increased range of up/down alignment assessments from 15 to 25 (in units of [ysamp] input spacing)
% - initial target batch selection updates:     
%   - [ops.targBatch] defines initial target batch for drift alignment
%     - default = 2/3;
%     - <1, target batch will be nearest `targBatch*100` percentile of total batches
%     - >=1, will be direct index to batch#
%     - output [rez.ops.targBatch] is updated to the **actual batch index** used
% - alignment weighting updates:
%   - initial batch is smoothed over window +/-5% of total batches
%   - scale breadth & smoothing applied to covariance estimates across rigid shift iterations
% 
% - extensive plotting updates
%   - plot each stage of evolution during rigid alignment estimation
%   - add shift estimate (red line) & target batch (cyan *) overlays to driftMap
%   - if drift correction is applied (do_correction>=0) && ops.fig>=2 (ks25 "debug plot" mode),
%     will also plot drift map readout **after** applying shifts to preprocessed data file
%     - confirm actual effect on data
%     - ...definite processing time impact, but important piece of information
% 
% - shift_batch_on_disk2.m updated to handle preprocessed file always starting at t=0
%   - data shift only applied to actual batch data
%   - **not t= (0:tstart) or (tend:inf)**
% 
% Minor:
% - excised dependence on [rez.temp] struct for Nbatch param
%   - unknown rationale for rez.temp duplicates in first place...
% 
% 
% ---
% 2021-xx-xx  TBC  Evolved from original Kilosort
% 2021-05-20  TBC  Cleaned & commented
% 2021-12-21  TBC  Respect threshold value in [ops.ThPre] (...was hardcoded ==10)
%


rez.iorig = 1:rez.ops.Nbatch;

if  getOr(rez.ops, 'nblocks', 1)==0
    do_correction = 0;
end

% record default in output
rez.ops.integerShifts = getOr(rez.ops, 'integerShifts', 0);

% display status of flag for whether to apply:
% - upsampled datashift (==1)
% - datashift rounded to interger channel spacing (==2),
% or no correction at all (just makes the datashift plots in that case)
% override with ops value, if exists
if do_correction
    % foolishly piggybacking on [related] input flag....
    do_correction = do_correction + rez.ops.integerShifts;
end

% flag for additional datashift plot of corrected data
debugPlot =  getOr(rez.ops, 'fig', 1)>=2;
[~, fname] = fileparts(rez.ops.saveDir); % name of output directory (for figure saving)

ops = rez.ops;

% The min and max of the y and x ranges of the channels
ymin = min(rez.yc);
ymax = max(rez.yc);
xmin = min(rez.xc);
xmax = max(rez.xc);

% Determine the average vertical spacing between channels. 
% Usually all the vertical spacings are the same, i.e. on Neuropixels probes. 
dmin = median(diff(unique(rez.yc)));
fprintf('Computing datashift alignment...\n')
fprintf('\tvertical pitch size is %d \n', dmin)
rez.ops.dmin = dmin;
rez.ops.yup = ymin:dmin/2:ymax; % centers of the upsampled y positions

% Determine the template spacings along the x dimension
% dminx = median(diff(unique(rez.xc)));
yunq = unique(rez.yc);
mxc = zeros(numel(yunq), 1);
for j = 1:numel(yunq)
    xc = rez.xc(rez.yc==yunq(j));
    if numel(xc)>1
       mxc(j) = median(diff(sort(xc))); 
    end
end
dminx = median(mxc);
fprintf('\thorizontal pitch size is %d \n', dminx)

rez.ops.dminx = dminx;
nx = round((xmax-xmin) / (dminx/2)) + 1;
rez.ops.xup = linspace(xmin, xmax, nx); % centers of the upsampled x positions

% ------------------------------------------------------------------
% upsampling bin width across Y (um)
% - 5 is default for 20um neuropixel spacing]; 
% - but 5 too small for uprobes, 10 good for 100 um uprobe spacing
dd = 10;  % [5]  

% - this should really be set as a function of y-spacing (dmin]
%   - rest of drift computation relies on indices of this sampling res.
%   - If dd res is too low/high for electrode spacing, drift fit will be suboptimal
% min and max for the range of depths
dmin = ymin - 1;
dmax  = 1 + ceil((ymax-dmin)/dd);


% [spkTh] "template amplitude" threshold for the generic templates & initial driftmap depth estimates
spkTh = floor(ops.ThPre);% [def=10]; % floor(ops.ThPre*1.25);
% ~!~ [spkTh] must be integer for use w/in standalone_detector.m ~!~


% Extract all the spikes across the recording that are captured by the
% generic templates. Very few real spikes are missed in this way. 
[st3, rez] = standalone_detector(rez, spkTh);
% NOTE:  [st3] output of  standalone_detector.m>>spikedetector3.cu  is *'int32'*
%  - thus st3(:,3), spike threshold variance explained, is [lossy] integer
%%

% detected depths
dep = st3(:,2);

% min and max for the range of depths
dmin = ymin - 1;
dep = dep - dmin;

dmax  = 1 + ceil(max(dep)/dd);
Nbatches      = rez.ops.Nbatch;
batchSec      = ops.NT/ops.fs;

% which batch each spike is coming from
batch_id = st3(:,5); %ceil(st3(:,1)/dt);

% preallocate matrix of counts with 20 bins, spaced logarithmically
nbins = 20;
ampRng = prctile(st3(:,3), [.5,99.5]);
binEdg = round(logspace(log10(ampRng(1)), log10(ampRng(2)), nbins+1));
% tune bin resolution to data
while any(diff(binEdg)<1)
    ampRng(2) = ampRng(2)+1;
    binEdg = round(logspace(log10(ampRng(1)), log10(ampRng(2)), nbins+1));
end
binEdg(end) = inf;

F = zeros(dmax, nbins, Nbatches);
for t = 1:Nbatches
    % find spikes in this batch
    ix = find(batch_id==t);
    
    % subtract offset
    dep = st3(ix,2) - dmin;
    
    % amplitude bin relative to range of spike amplitudes detected in data    
    [~, ~, amp] = histcounts(st3(ix,3), binEdg);
    
    % distribute into bins with sparse trick
    % sparse is very useful here to do this binning quickly
    M = sparse(ceil(dep/dd), amp, ones(numel(ix), 1), dmax, nbins);
    
    % the counts themselves are taken on a logarithmic scale (some neurons
    % fire too much!)
    F(:, :, t) = log2(1+M);
end

%%
% determine registration offsets (output in [imin])
ysamp = dmin + dd * [1:dmax] - dd/2;
[imin,yblk, F0, F0m, rez.ops.targBatch] = align_block2(F, ysamp, ops);

if isfield(rez, 'F0')
    d0 = align_pairs(rez.F0, F0);
    % concatenate the shifts
    imin = imin - d0;
end

% convert shift to um 
dshift = imin * dd;

if rez.ops.integerShifts
    % keep record of full-res dshifts for good measure
    rez.dshift0 = dshift;
    % smooth dshift to reduce flip-flop jumps
    dshift = smoothdata(dshift, 'movmean',6);
    % round shifts to integers of channel spacing
    dmin = median(diff(unique(rez.yc)));
    dshift = round(dshift./dmin).*dmin;
end

% keep track of dshift 
rez.dshift = dshift;


%%
if getOr(ops, 'fig', 1)  
    figure;
    set(gcf, 'Color', 'w')
    xl = [-.5,Nbatches+0.5]*batchSec + ops.trange(1); % xlim in seconds
    
    % plot the shift trace in um
    plot(rez.dshift,'-r')
    llabl = {'rez.dshift'};
    box off
    if isfield(rez,'dshift0')
        hold on
        plot(rez.dshift0,'color',[.7 .7 1]);
        llabl = [llabl, 'rez.dshift0 (full-res)'];
    end
    xlabel('batch number')
    ylabel('drift (um)')
    title('Estimated drift traces')
    legend(llabl)
    drawnow
    
    % raster plot of all spikes at their original depths
    ii = st3(:,3)>=spkTh;
    st_depth0 = st3(ii,2);
    st_depthD = st_depth0 + imin(batch_id(ii)) * dd;
    xs = (.5:1:Nbatches)*batchSec + ops.trange(1);
    
    H = figure;
    set(H, 'name', ['driftMap_',fname])
    nsubp = 2+1*(do_correction>0);
    hax1 = subplot(nsubp,1,1); hold on; box off
    hax2 = subplot(nsubp,1,2); hold on; box off
    
    ampClim = round(prctile(st3(:,3), [.1 99.9]));
    ampClim(2) = max(ampClim(2),50); % reveal low amplitudes
%     ampCmax = round(max(50, ampMax*0.65));
    climLabl  = sprintf('\t\tamplitude clim = [%d, %d]', ampClim);
    for j = spkTh:max(st3(:, 3))
        % for each amplitude bin, plot all the spikes of that size in the
        % same shade of gray
        ix = st3(:, 3)==j; % the amplitudes are rounded to integers
        thisCol = [1 1 1] * max(0, 1-(j-spkTh/2)/ampClim(2)); % scale datapoint color range
        plot(hax1, st3(ix, 1)/ops.fs, st_depth0(ix), '.', 'color', thisCol) % the marker color here has been carefully tuned
        plot(hax2, st3(ix, 1)/ops.fs, st_depthD(ix), '.', 'color', thisCol) % the marker color here has been carefully tuned
    end
    axis tight
    xlim(hax1, xl);
    linkaxes([hax1,hax2]);
    % overlay drift correction estimate on initial map
    plot(hax1, xs, imin * dd +diff(ylim(hax2))/2, '-r');
    B = rez.ops.targBatch;
    plot(hax1, xs(B), imin(B) * dd +diff(ylim(hax2))/2, '*c');
    % add channel numbers to yaxis
    ych = floor(linspace(1,length(rez.yc),9));
    text(hax1, xl(2)*ones(9,1), rez.yc(ych), num2str(ych(:)), 'color','c', 'VerticalAlignment','middle', 'FontSize',10)

    title(hax1, 'Drift map: Initial', 'interp','none')
    ylabel(hax1, 'Init spike position (um)')
    xlh = xlabel(hax1, climLabl, 'HorizontalAlignment','left','fontsize',8);
    set(xlh, 'position', get(xlh,'position').*[0,1,1]);
    title(hax2, 'Drift map: Corrected', 'interp','none')    
    ylabel(hax2, 'Shifted spike position (um)')
    xlabel(hax2, 'time (sec)')

    % name figure
    addFigInfo(ops, H);

end
%%

% NO@!!@  Please stop messing with time for drift correction!!!
% % this is not really used any more, should get taken out eventually
% [~, rez.iorig] = sort(mean(dshift, 2));

    
if do_correction>=1
    cmdLog(sprintf('Applying drift correction to\t%s', ops.fproc));
    if rez.ops.integerShifts
        % minimize sigma for the Gaussian process smoothing
        sig = 1; % no smoothing across channels (else produces unwanted a gain reduction on probes w/site spacing >= 50um)
    else
        % sigma for the Gaussian process smoothing
        sig = rez.ops.sig;        
    end
    % open destination file for WRITING
    fdest = fopen(ops.fproc, 'r+');
    % register the data batch by batch
    %     dprev = gpuArray.zeros(ops.ntbuff,ops.Nchan, 'single');
    for ibatch = 1:Nbatches
        %         dprev = shift_batch_on_disk2(rez, ibatch, dshift(ibatch, :), yblk, sig, dprev);
        % ks25: shift_batch_on_disk2.m updated to handle preprocessed file always starting at t=0
        %  - data shift only applied to actual batch data, **not t= (0:tstart) or (tend:inf)**
        shift_batch_on_disk2(rez, ibatch, dshift(ibatch, :), yblk, sig, fdest);
    end
    % close file
    fclose(fdest);
    cmdLog(sprintf('Shifted up/down %d batches.', Nbatches), toc);
else
    if getOr(ops,'fig',1)
        ht = get(hax2, 'title');
        set(ht, 'string', [get(ht,'string'),'  (** not applied to data [per ops flag])']);
    end
    cmdLog(sprintf('%s drift correction computed, but not applied.',mfilename), toc);
end

% keep track of original spikes
rez.st0 = st3;

rez.F = F;
rez.F0 = F0;
rez.F0m = F0m;


    
% next, we can just run a normal spike sorter, like Kilosort1, and forget about the transformation that has happened in here 


% % % % %
%% additional debug plotting
% - read back the newly shifted data and plot driftmap
% - should look just like shifted driftmap, but with clipping on any channels that were shifted out of dat file register
% 
% - slow & just for confirmation of what the ACTUAL saved whitened data looks like

if debugPlot % && any(rez.dshift)
    if (~any(rez.dshift) || do_correction<=0) 
        if nsubp>2
            hax3 = subplot(nsubp,1,3); hold on; box off
            text(0,0,'( drift correction not applied to data [per ops flag] )', 'fontsize',12, 'FontAngle','italic', 'HorizontalAlignment','center');
            axis(hax3, [-1 1 -1 1]);
            axis(hax3, 'off');
        end
    else
        %% One more extraction & plot to show actual shifts applied
        % This is slow, but reassuring.  ....disable when certain of parameters
        
        % use same spike threshold as used initially spkTh = ops.ThPre; %10;
        
        % Extract all the spikes across the recording that are captured by the
        % generic templates. Very few real spikes are missed in this way.
        % - don't pass out rez struct this time
        % - we don't want/need any of this to carry over, just checking our work
        [st3] = standalone_detector(rez, spkTh);
        
        % detected depths
        dep = st3(:,2);
        
        % min and max for the range of depths
        dmin = ymin - 1;
        dep = dep - dmin;
        
        dmax  = 1 + ceil(max(dep)/dd);
        Nbatches      = rez.ops.Nbatch;
        batchSec      = ops.NT/ops.fs;
        
        % which batch each spike is coming from
        batch_id = st3(:,5); %ceil(st3(:,1)/dt);

        % Apply same amplitude binning as before
        F = zeros(dmax, nbins, Nbatches);
        for t = 1:Nbatches
            % find spikes in this batch
            ix = find(batch_id==t);
            dep = st3(ix,2) - dmin;
            [~, ~, amp] = histcounts(st3(ix,3), binEdg);
            M = sparse(ceil(dep/dd), amp, ones(numel(ix), 1), dmax, nbins);
            F(:, :, t) = log2(1+M);
        end

        figure(H)
        % raster plot of all spikes at their original depths
        ii = st3(:,3)>=spkTh;
        st_depth0 = st3(ii,2);
        
        hax3 = subplot(nsubp,1,3); hold on; box off
        
        for j = spkTh:100
            % for each amplitude bin, plot all the spikes of that size in the
            % same shade of gray
            ix = st3(:, 3)==j; % the amplitudes are rounded to integers
            thisCol = [1 1 1] * max(0, 1-j/60);
            plot(hax3, st3(ix, 1)/ops.fs, st_depth0(ix), '.', 'color', thisCol) % the marker color here has been carefully tuned
        end
        axis tight
        linkaxes([hax1,hax2,hax3]);
        plot(hax3, xs, rez.dshift +diff(ylim(hax2))/2, '-r');
        
        title(hax3, 'Drift map: Applied', 'interp','none')
        ylabel(hax3, 'Drifted spike position (um)')
        xlabel(hax3, 'time (sec)')
    end
end

%% save figure
% if [figDir] variable with full path to figure destination exists in the base workspace,
% save figure as png
if getOr(ops, 'fig', 1) && evalin('base','exist(''figDir'',''var'');')
    % save drift figure as png (all other formats result in excessively large files)
    figDir = evalin('base','figDir');
    figureFS(H, 'portrait', 15*[1 1]); % standardize figure output
    try
        saveFigTriplet(1, [], 'png');
    catch
        fname = get(H, 'name');
        fname(fname==32 | fname==44 | fname==filesep)='_'; % no spaces, commas, or [filesep] in figure name!
        fname = [fname,datestr(now, '_yyyymmmdd')];
        figDir = fullfile(figDir,'png');
        if ~exist(figDir,'folder')
            mkdir(figDir)
        end
        saveas(H, fullfile(figDir,[fname,'.png']),'png')
    end
end


end % main function

