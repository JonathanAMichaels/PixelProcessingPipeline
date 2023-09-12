function hout = figureFS(h, ori, type)
% function h = figureFS(h, ori, type)
% 
% Create and/or update figure for saving ("FS")
% - Adjust figure settings for consistent print/save outputs, and [most of the time]
%   produce an upright image regardless of save format
%   - in certain cases output ori does fail, but fails without clipping content
%
% INPUTS:
%   h       = handle to figure (non destructive). Opens new figure by default.
%   ori     = paper orientation:  'portrait' or 'landscape'(def)
%   type    = paper type: string 'A4', 'B5', 'usletter', 'tabloid'(def), etc.
%               (...or width x height in inches)
% OUTPUTS:
%   hout    = (optional) handle to figure
%
% %---
% %Basic usage:
%   % open figure #99, and set to A5 paper size in portrait ori
%   figureFS(99, 'portrait', 'a5');
%
% %---
% %Advanced usage:
%   H = figure;
%   % create figure with lots of subplots
%   spx = 15;  spy = 2;
%   for i = 1:spx
%       subplot(spy, spx, i);
%       h = plot(randn(50,1), randn(50,1), '.');
%       subplot(spy, spx, i+spx);
%       histogram(get(h, 'Xdata'));
%   end
%   % Now, update figure to fit contents:
%   % - apply name, set custom paper size for subplots, [default to landscape orientation]
%   figureFS( {H, 'lotsOfSubplots'}, [], [2*spy,2*spx] );
%   saveas(H, [get(gcf,'name'), datestr(now,'_yyyymmmdd_HH-MM-ss')], 'pdf')
% %---
% 
% 2012-12-09  TBC  Wrote it. (Finally!)
% 2014-07-16  TBC  Updates & input info
% 2021-05-15  TBC  Cleaned & commented
%

% NOTE: Rasterized formats (jpg, png, tiff) *consistently* come out rotated
% by 90 deg, but content spacing and aspect ratio are correct & unclipped (!?@!!)
% Vector formats (pdf & eps) of the same figures are oriented correctly...go figure.
% -- TBC

if ~exist('h','var')
    h = figure;
elseif isempty(h)
    h = figure;
elseif iscell(h)
    if isempty(h{1})
        h{1} = figure;
    else
        figure(h{1});
    end
    set(h{1}, 'name',h{2});
    h = h{1};
elseif ischar(h)
    nm = h;
    h = figure;
    set(h, 'name',nm);
else
    figure(h)
end

if ~exist('ori','var') || isempty(ori)
    ori = 'landscape';
end

if ~exist('type','var') || isempty(type)
    type = 'tabloid';
end


set(h,'PaperOrientation', ori);
    orient(ori); %...and again (thanks Matlab)
    
try
    set(h,'PaperType', type);
catch
    set(h, 'PaperType','<custom>', 'PaperSize', sort(type,'descend'))
end


set(h,'PaperUnits', 'inches')
sz = get(h,'PaperSize');
set(h,'PaperPositionMode','manual');
n = min([.1*sz, 0.5]);
set(h,'PaperPosition',[n/2, n/2, sz(2)-n, sz(1)-n]);

% just keep doing this till it sticks
set(h,'PaperOrientation', ori);
orient(h, ori); %...and again (thanks Matlab)

% silence unneeded outputs
if nargout>0
    hout = h;
end

end %main function
