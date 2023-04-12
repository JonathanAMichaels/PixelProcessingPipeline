function plotTemplateDynamics(rez, theseUnits)
% function plotTemplateDynamics(rez, theseUnits)
%
% Scrappy function to visualize changes in template dynamics across spike extraction.
% - Produces image plots of temporal (W) & spatial (U) templates across each batch of extraction
% - Overlays number of spikes extracted in each batch
% 
% INPUTS:
%   rez         = standard rez struct from Kilosort session (must be from [ks25] codebase; must include .WA & .UA)
%   theseUnits  = indices of which units to plot
%                 - if none provided, will randomly select 24 templates to plot
%                 - if theseUnits=='all', will plot all units (...not recommended for >=40 total units)
% ---
% EXAMPLE
%   - run ks25 sort from GUI interface
%   - [ks] (a handle to the kilosort object) should be created in base workspace
%   - from command window:
%       plotTemplateDynamics(ks.rez, ks.rez.troubleUnits)
% ---
% 2021-xx-xx  TBC  Wrote it.
% 

sz = size(rez.WA);
if isempty(rez.WA) || numel(sz)<3 
    fprintf(2, '\tNo record of template dynamics in this rez struct\n')
    return
end

if nargin>1
    if strcmp(theseUnits,'all')
        theseUnits = 1:sz(2);
    end
    nplots = length(theseUnits)
else
    nplots = 24;
end
spx = ceil(sqrt(nplots));
spy = ceil(sqrt(nplots));

if ~exist('theseUnits','var') || isempty(theseUnits)
    theseUnits = sort(randperm(sz(2),nplots));
end

iPC = [1,2]; % which pc to plot

for k = 1:length(iPC)
    H = figure;
    set(H, 'name',sprintf('PC%d',iPC(k)));
    
    for i = 1:nplots
        u = theseUnits(i);
        
        subplot(spx, spy, i)
        imagesc( sq(rez.WA(:, u, iPC(k), :)) );
        box off
        title(sprintf('unit % 3d || % 3d',u, rez.nsp(u) ));
        hold on
        if isfield(rez,'invDetected') && any(rez.invDetected(u,:))
            plot(find(rez.invDetected(u,:)), 5, '.r','markersize',5);
        end
        if isfield(rez,'nspA')
            yyaxis right
            hl = plot(rez.nspA(u,:)','.');
            set(hl.MarkerHandle, 'Style','hbar', 'size',3);
            ylabel('spikes added','fontsize',8)
        end
        
        
    end
    
    addFigInfo(rez.ops, H)
end

iUA = [1,2,3]; % which UA weight to plot

for k = 1:length(iUA)
    H = figure;
    set(H, 'name',sprintf('UA-%d',iUA(k)));
    
    for i = 1:nplots
        u = theseUnits(i);
        
        subplot(spx, spy, i)
        imagesc( sq(rez.UA(:, u, iUA(k), :)) );
        box off
        title(sprintf('unit % 3d || % 3d',u, rez.nsp(u) ));
        set(gca, 'clim',[-.2,1])
        hold on
        if isfield(rez,'invDetected') && any(rez.invDetected(u,:))
            plot(find(rez.invDetected(u,:)), 5, '.r','markersize',5);
        end
        if isfield(rez,'nspA')
            yyaxis right
            hl = plot(rez.nspA(u,:)','.');
            set(hl.MarkerHandle, 'Style','hbar', 'size',3);
            ylabel('spikes added','fontsize',8)
        end
        
        
    end
    
    addFigInfo(rez.ops, H)
end