function saveFigTriplet(withdate, infostr, fileflags, figSubDir, varargin)
% function saveFigTriplet(withdate, infostr, fileflags, figSubDir, varargin)
%   fileflags is logical array for saving:  [tiff, png, eps, mat, pdf]
% 
% Saves current figure as .fig, .pdf, and .eps (by default)
%   3rd input selects filetypes as logical index:  [tiff, png, eps, mat, pdf]
%   File name comes from figure name:   get(gcf, 'name')
%   --by default adds current date to file name
%   Mines calling workspace for path to figure directory ('figDir')
%   --creates dir structure within figDir if not already present
%
%   
% 2014-04-16 TBC  Wrote it. (czuba@utexas.edu)  
% 2021-05-18 TBC  Cleaned & commented
%

% NOTE: Rasterized formats (jpg, png, tiff) *consistently* come out rotated
% by 90 deg, but content spacing and aspect ratio are correct & unclipped (!?@!!)
% Vector formats (pdf & eps) of the same figures are oriented correctly...go figure.
% -- TBC

H = gcf;

%% Parse inputs
if ~exist('withdate','var') || isempty(withdate)
    % default: append date to filename as '_yyyymmmdd'
    withdate = 1;
end

% file save types (default [.mat, .pdf, .eps])  ...convoluted, but backwards compatible and extensible
filetypes = struct('tiff',0, 'png',0, 'eps',1, 'matlab',1, 'pdf',1);
filefn = fieldnames(filetypes);

if nargin>2 && ~isempty(fileflags)
    if ischar(fileflags)
        fileflags = {fileflags};
    end
    if iscell(fileflags)
        % string input
        for i = 1:length(filefn)
            filetypes.(filefn{i}) = contains(filefn{i},fileflags);
        end
    else
        for i = 1:length(fileflags)
            filetypes.(filefn{i}) = fileflags(i);
        end
    end
end

if nargin <4 || isempty(figSubDir)
    figSubDir = [];
end

% get vars from figure tag or caller wkspc
Htag = get(H, 'tag');
if ~isempty(Htag) && contains(Htag, filesep)
    figDir = Htag;
    
elseif evalin('caller','exist(''figDir'',''var'')')
    figDir = evalin('caller','figDir');    
end


%% Determine filename
fname = get(H,'name');
if withdate
    datefmt = 'yyyymmmdd';
    if withdate>1
        datefmt = [datefmt,'-HHMMSS'];
    end
    fname = [fname,'_',datestr(now, datefmt)];
end

fname(fname==32)='_'; % no spaces in figure name!
fname(fname==44)='_'; % no commas either!

% default name as date w/time to prevent crash for saving w/o name
if isempty(fname)
    fname = datestr(now,'yyyymmmdd_HH-MM-ss');
end
    
if ~exist('figDir','var') || isempty(figDir)
    figDir = fullfile(pwd,'figs');
end


%% Append info string to bottom of figure
%  - if provided as input or if exist in the workspace
ax = gca;
fsz = 10;
% check for infostr in calling directory if not passed as input
if ~exist('infostr','var') || isempty(infostr)
    if evalin('caller','exist(''infostr'',''var'')')
        infostr = evalin('caller','infostr');
    end
end
% now apply if one exists
if exist('infostr','var') && ~isempty(infostr)
    axes('position',[0,.002,1,.02],'visible','off');
        % shrinking text if multiple lines
        if contains(infostr, {sprintf('\n'),sprintf('\n\r')}), fsz = 8; end
    text(0,0, infostr, 'verticalAlignment','bottom', 'interpreter','none', 'fontsize',fsz);    
end
axes(ax);


%% Create dest directories if don't already exist
t = logical(struct2array(filetypes));   filefn = fieldnames(filetypes);
t = filefn(t);

% Make dirs
for i = 1:length(t)
    if ~exist(fullfile(figDir,t{i}),'dir')
        mkdir(fullfile(figDir,t{i}));
    end
    if ~isempty('figSubDir') && ~exist(fullfile(figDir,t{i},figSubDir),'dir')
        mkdir(fullfile(figDir,t{i},figSubDir));
    end
end


%% Do the saving

% Matlab figure
if filetypes.matlab
    savefig(H, fullfile(figDir,'matlab',figSubDir,[fname,'.fig']), 'compact') % smaller
end

if filetypes.tiff
    saveas(H,fullfile(figDir,'tiff',figSubDir,[fname,'.tif']),'tiff')
end

% PNG
if filetypes.png
    saveas(H,fullfile(figDir,'png',figSubDir,[fname,'.png']),'png')
end


% PDF
if filetypes.pdf
    try
        eval(['print ','-dpdf -r400',' -painters ',fullfile(figDir,'pdf',figSubDir,[fname,'.','pdf'])]);
        %         eval(['print ','-dpdf -r400',' -opengl ',fullfile(figDir,'pdf',[fname,'.','pdf'])]);
        catch,    fprintf(2,'\t\tErrored trying to save %s file: %s\n', 'pdf', fname);
    end
end

% EPS
if filetypes.eps
    try
        eval(['print ','-depsc',' -painters ',fullfile(figDir,'eps',figSubDir,[fname,'.','eps'])]);
        %         eval(['print ','-depsc',' -opengl ',fullfile(figDir,'eps',[fname,'.','eps'])]);
        catch,    fprintf(2,'\t\tErrored trying to save %s file: %s\n', 'eps', fname);
    end
end

% Tell em what you did
fprintf(2, '\b\tSaved to: %s\n ', figDir);


end %main function
