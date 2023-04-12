function cm = createChannelMapFile(nChan, ysp, xsp, chPerRow, varargin)
% function cm = createChannelMapFile(nChan, ysp, xsp, chPerRow, varargin)
%
% Functional version of Kilosort channel map creation
% - chanMap file will be saved in default kilosort configFiles directory
%   as <cm.name>.mat    (...fallback to [pwd], asks to overwrite if file already exists)
% - include or specify any additional channel map fields as additional PV pair input:  ...'parameter', value...)
% - ks25 updated to retain any additional recording device info the user chooses to include in chanMap file
% 
% INPUTS:
%   nChan   = total number of channels
%   ysp     = y spacing [um]
%   xsp     = x spacing [um]
%   chPerRow    = number of channels per row
%
%   additional/optional PV pairs:
%       'name'      (def = 'example<nChan>ch<fs/1000>k')
%       'fs'        (def = 40000)
%       'configFxn' (def = '')
% 
% OUTPUTS:
%   cm  = channel map struct
% 
% Channel numbering & spacing convention:
% - channel numbers increase from left-to-right, proximal-to-distal
% - y-zero at most distal channel
% - y-coord advances in ysp increments up the probe
%   - such that y-zero coincides with tareing the probe depth
%     at first detectable entry into brain
%   - subsequent channel depths are then:  probe depth - y-coord
% - x-coord are balanced on either side of x-zero
%  
% ---
% 2021-06-16  TBC  Wrote it.
% 

% Setup defaults
if nargin<1 || isempty(nChan)
    % channel count
    nChan = 32;
end
if nargin<2 || isempty(ysp)
    % y spacing (across probe length)
    ysp = 100;
end
if nargin<3 || isempty(xsp)
    % x spacing (across probe width)
    xsp = 50;
end
if nargin<4 || isempty(chPerRow)
    % channels per row:
    % 1==linear array, 2==stereotrode, ...
    chPerRow = 2; % 'stereotrode'
end

defFreq = 40000;

% Parse PV pairs
pp = inputParser();
pp.addParameter('name','');         % channel map name
pp.addParameter('fs',40000);        % sampling frequency
pp.addParameter('configFxn','');    % function handle to kilosort config function
pp.KeepUnmatched = 1;               % allow additional user parameters
pp.parse(varargin{:});
argin = pp.Results;

% incorporate additional inputs (...I swear this used to happen automatically)
if ~isempty(pp.Unmatched)
    for uf = fieldnames(pp.Unmatched)'  % counter cheat
        argin.(uf{:}) = pp.Unmatched.(uf{:});
    end
end
            
if isempty(argin.name)
    % apply outside of parser so input fields can be used in name
    argin.name =  sprintf('example%dch%dk', nChan, round(argin.fs/1000));
end

ksRoot = fileparts(which('kilosort'));
if ~isempty(ksRoot)
    configDir = fullfile(ksRoot, 'configFiles');
else
    configDir = pwd;
end


%% chanMap coordinates
%     nChan = 32; ysp = 100; xsp = 50;
nRows = nChan/chPerRow;
yc = ((nRows-1)*ysp):-ysp:0;
yc = kron( yc, ones([1,chPerRow]))';
xc = xsp*(0:chPerRow-1)
xc = repmat( xc-mean(xc), [1,nRows])';


%% create chanMap struct
cm = argin; % .name, .freq, .configFxn, and whatever other PV pairs user chooses to include
cm.chanMap = (1:nChan)';
cm.chanMap0ind = cm.chanMap-1;
if ~isfield(cm,'connected')
    cm.connected = true(size(cm.chanMap));
end
cm.xcoords = xc;  % um
cm.ycoords = yc;  % um


%% save to kilosort root dir
fname = fullfile(configDir, [cm.name, '.mat']);
if exist(fname,'file')
    [fname, fpath] = uiputfile('*.mat', 'chanMap file already exists, adjust or overwrite', fname);
    fname = fullfile(fpath, fname);
end
save(fname, '-struct', 'cm');


end %main function

% -------------------------------------------------------------------------------
% -------------------------------------------------------------------------------
%% Original Kilosort chanMap setups
% -------------------------------------------
%     %  create a linear 32 channel map file
%     Nchannels = 32;
%     connected = true(Nchannels, 1);
%     chanMap   = 1:Nchannels;
%     chanMap0ind = chanMap - 1;
%     xcoords   = ones(Nchannels,1);
%     ycoords   = [1:Nchannels]';
%     kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)
% 
%     fs = 25000; % sampling frequency
%     save('C:\DATA\Spikes\20150601_chan32_4_900s\chanMap.mat', ...
%         'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
% 
% -------------------------------------------
%     %  create a tetrode-row 32 channel map file
%     Nchannels = 32;
%     connected = true(Nchannels, 1);
%     chanMap   = 1:Nchannels;
%     chanMap0ind = chanMap - 1;
% 
%     xcoords   = repmat([1 2 3 4]', 1, Nchannels/4);
%     xcoords   = xcoords(:);
%     ycoords   = repmat(1:Nchannels/4, 4, 1);
%     ycoords   = ycoords(:);
%     kcoords   = ones(Nchannels,1); % grouping of channels (i.e. tetrode groups)
% 
%     fs = 25000; % sampling frequency
% 
%     save('C:\DATA\Spikes\Piroska\chanMap.mat', ...
%         'chanMap','connected', 'xcoords', 'ycoords', 'kcoords', 'chanMap0ind', 'fs')
% -------------------------------------------

