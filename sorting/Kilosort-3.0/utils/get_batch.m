function dat = get_batch(ops, ibatch, fid, varargin)
% function dat = get_batch(ops, ibatch, fid, varargin)
% 
% Retrieve buffered batch of data from preprocessed data file [ops.fproc],
% return as gpuArray.
% 
% Unified version of original Kilosort get_batch.m for either standard file reads
% or memory mapped files. Defaults to ops.useMemMapping = 1
% 
% 
% INPUTS:
%   [ops]       standard kilosort ops struct
%   [ibatch]    1-based index of batch to load
%   --optional--
%   [fid]       handle to memmappedfile opject or to standard fopen file
%   [flags]     string flags for modifying batch features:  'nobuffer', 'noscale' (see below)
% 
% OUTPUTS:
%   [dat]       gpuArray of batch data, sized [nsamples-by-nchannels]
%               - loaded directly to a gpuArray, after converting to single & scaling by ops.scaleproc
%               - NOTE: dat orientation is transposed relative to saved file (for efficient usage w/in GPU code)
% 
% [fid] options:
% - if is handle to a Memory Mapped file, load batch via memmapping
% - if is index to standard open file, load batch via standard reads
% - if [fid] absent or empty, follow ops.useMemMapping flag, open/initialize as needed
%   (def: useMemMapping = 1)
%   - if useMemMapping, will check for memmappedfile object in standard location:  ops.fprocmmf
%   - if ~useMemMapping, will fopen & fclose access to:  ops.fproc
% 
% varargin accepts modifier string flag(s):
%   'nobuffer'  don't add buffer to either side of batch indices; size(dat)==[nsamp,nchan]
%   'noscale'   don't scale data by ops.scaleproc
% 
% If using memory mapped reads, must follow convention of preprocessDataSub.m
%   - handle to memmapfile object must be in:  ops.fprocmmf
%   - .Data field must be:  '.chXsamp'
%   (i.e.  rez.ops.fprocmmf = memmapfile(filename, 'Format',{datatype, [chInFile nSamp], 'chXsamp'});
%   - in theory, memory mapped reads *might* work without having set up any .fprocmmf handle in advance,
%     but good chance it would cause a mess & be very slow in the least.
%     ...best bet is to properly setup use of ops.useMemMapping = 1 from the get go
% 
% ---
% 2021-04-13  TBC  Updated with proper usage of .ntbuff
%                  Sends data directly to gpuArray
%                  memory mapped version
% 2021-04-20  TBC  Unified version determines read method based on [fid] input type or ops.useMemMapping
% 2021-04-28  TBC  Cleaned & commented.
% 

%% parse inputs & determine read method from inputs

useMemMapping = getOr(ops, 'useMemMapping', 1);
cleanupFid  = 0; % flag to close fid, if opened w/in this function

if nargin<3 || isempty(fid)
    if useMemMapping
        if isfield(ops, 'fprocmmf') && ~isempty(ops.fprocmmf)
            % look for memmapped handle in ops struct
            mmf = ops.fprocmmf;
        else
            % prob slow on-the-fly, should pass this as input (ops.fprocmmf) whenever feasible
            filename    = ops.fproc;
            chInFile    = ops.NchanTOT;
            nSamp       = ops.tend;
            datatype    = 'int16';
            mmf         = memmapfile(filename, 'Format',{datatype, [chInFile nSamp], 'chXsamp'});
        end
    else
        % open preprocessed data file for standard reads
        fid = fopen(ops.fproc, 'r');
        cleanupFid = 1;
    end
    
elseif isa(fid, 'memmapfile')
    % input is memmapped file handle
    mmf = fid;
    useMemMapping = 1;
    
elseif ~isempty(fid)
    %`standard reads from [fid] input
    useMemMapping = 0;
end


%% Define batch start, size, & padding (if necessary)

% check for [varargin] flags
useBuffer   = ~any(contains(varargin,'nobuffer', 'IgnoreCase',1));

if ~any(contains(varargin,'noscale', 'IgnoreCase',1))
    scaleDat = ops.scaleproc; % scale integer data on load 
else
    scaleDat = 1;
end

NT          = ops.NT;       % samples per batch
ntbuff      = ops.ntbuff * useBuffer;       % single-end buffer size (samples)
NTwin       = [-ntbuff, NT+ntbuff-1]; % first & last sample of this buffered batch
        % iNTbuff = -ntbuff:1:(NT+ntbuff-1); % == .NT samples padded w/ .ntbuff on either side
        
tstart  = ops.tstart;    % accomodate non-zero first sample time
tend    = ops.tend;

% starting sample offset for this batch (in samples; 1-based index)
offset = 1 + tstart + NT*(ibatch-1);

% sample window
sampWin = NTwin + offset;

% prepad error check
if sampWin(1)<1 % any <=0
    prepad = -sampWin(1) + 1;
    sampWin(1) = 1;
else
    prepad = 0;
end

% postpad error check
if sampWin(2)>tend % any <=0
    postpad = sampWin(2) - tend;
    sampWin(2) = tend;
else
    postpad = 0;
end


%% Read data directly to GPU
% - transpose for gpu (gpu orientation ~= dat orientation)
% - convert to singles
% - scale by [ops.scaleproc]

if useMemMapping
    % Read mmf data directly to GPU
    dat = gpuArray( single(mmf.Data.chXsamp(:, sampWin(1):sampWin(2))') / scaleDat);

else
    % adapt for fseek bytes & fread inputs
    % go to starting point (in bytes)
    fseek(fid, 2*(ops.Nchan * (sampWin(1)-1)), 'bof');
    dat = gpuArray( single( fread(fid, [ops.Nchan, diff(sampWin)], '*int16')') / scaleDat);

    % close file if opened w/in this function
    if cleanupFid
        % clean up
        fclose(fid);
    end 
end


%% Pad as necessary (***gpu orientation==[samples, channels]***)
dnom = 3; % rate of padding exponential decay
if prepad
    % smooth padding
    dat = [dat(1,:) .* exp(linspace(-prepad/dnom,0,prepad+1))'; dat(2:end, :)];
end

if postpad
    % smooth padding
    dat = [dat(1:end-1,:); dat(end,:) .* exp(linspace(0,-postpad/dnom,postpad+1))'];
end


end %main function
