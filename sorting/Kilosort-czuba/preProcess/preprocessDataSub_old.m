function rez = preprocessDataSub(ops)
% this function takes an ops struct, which contains all the Kilosort2 settings and file paths
% and creates a new binary file of preprocessed data, logging new variables into rez.
% The following steps are applied:
% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values
% 
% [ks25] updates:
% - adds git tracking with complete status, revisions, & changes to kilosort repo
% - uses memory mapped file reads by default (much faster)
% - updated to "_faster" version of get_whitening_matrix (memmapped reads AND parallelized loading...mmmuch faster)
% - creates handle to memmapped preprocessed data file in:  rez.ops.fprocmmf
% - disabled [linear] weighted smoothing of batches
%   - seems unnecessary & potentially problematic
%   - esp in cases where batch buffer [.ntbuff] is significantly longer than waveform length [.nt0]
% - removed creation of rez.temp (unclear why this existed in first place)
%   - required replacing instances of [rez.temp] to normal [rez] struct throughout codebase
% 
% ---
% 202x-xx-xx  TBC  Evolved from original Kilosort
% 2021-04-28  TBC  Cleaned & commented
% 

% track git repo(s) with new utility (see <kilosortBasePath>/utils/gitStatus.m)
if getOr(ops, 'useGit', 1)
    ops = gitStatus(ops);
end

tic;
ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned

NT       = ops.NT ; % number of timepoints per batch
NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc

bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /NT); % number of data batches
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault); % if .NchanTOT was left empty, then overwrite with n channels in file

ops.igood = true(size(chanMap));

ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;

NTbuff      = NT + 2*ops.ntbuff;
%  NOTE: formerly "+ 3*ops.ntbuff" when linearly weighting current batch with previously processed copy of itself

rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;

doCAR = getOr(ops, 'CAR', 1); % demean raw dat within channel with common *median* referencing


% set up the parameters of the filter % just one instance of filters
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass'); % butterworth filter with only 3 nodes (otherwise it's unstable for float32)
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high'); % the default is to only do high-pass filtering at 150Hz
end

cmdLog('Computing whitening matrix...', toc);
% fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
Wrot = get_whitening_matrix_faster(rez); % outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance of the data
% Wrot = gpuArray.eye(size(Wrot,1), 'single');
% Wrot = diag(Wrot);

cmdLog('Loading raw data and applying filters...', toc);
% fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r'); % open for reading raw data
if fid<3
    error('Could not open %s for reading.',ops.fbinary);
end

fidW        = fopen(ops.fproc,   'wb+'); % open for writing processed data
if fidW<3
    error('Could not open %s for writing.',ops.fproc);    
end

% weights to combine batches at the edge
% - WAIT, WHAAAAAT????
w_edge = linspace(0, 1, ops.ntbuff)';
ntb = ops.ntbuff;
datr_prev = gpuArray.zeros(ntb, ops.Nchan, 'single');

% scrappy progress bar in command window
allBatches = 1:Nbatch;
pb = progBar(allBatches, 20);

for ibatch = allBatches
    % we'll create a preprocessed binary file by reading from the raw file (NOT memory mapped) batches of NT samples,
    % with each batch padded by ops.ntbuff samples from before & after, to have as buffers for filtering
    offset = max(0, ops.twind + 2*NchanTOT*(NT * (ibatch-1) - ntb)); % number of BYTES to offset start of standard read operation
    fseek(fid, offset, 'bof'); % fseek to batch start in raw file
    buff = fread(fid, [NchanTOT NTbuff], '*int16'); % read and reshape. Assumes int16 data (which should perhaps change to an option)
    
    if isempty(buff)
        break; % this shouldn't really happen, unless we counted data batches wrong
    else
        nsampcurr = size(buff,2); % how many time samples the current batch has
    end
    
    % % ---------------------------------------------------------------------------------------------------------------
    % % --- step inside gpufilter.m operations ---%
    % - unpacked this utility function b/c unhappy with buffer padding before demeaning 
    
    % subsample only good channels & transpose for filtering
    datr = single(buff(chanMap,:))';    % buff dims now: [samples, channel]

    % --- Demean before padding ---
    % subtract within-channel means from each channel
    datr = datr - mean(datr, 1); % subtract mean of each channel
    
    % CAR, common average referencing by median
    if doCAR
        datr = datr - median(datr, 2); % subtract median across channels
    end
    
    % Now can pad first & last batches with zeros
    if nsampcurr<NTbuff
        if ibatch~=1
            % append zeros if end batch
            datr = [datr; zeros(NTbuff-nsampcurr, size(datr,2))];
        else
            % prepend zeros if first batch
            datr = [zeros(NTbuff-nsampcurr, size(datr,2)); datr];
        end
        % Kilosort [upstream] switched to padding with first/last sample
        % - not sure this is actually better if we're demeaning first (...it could DEF skew if not)
        % buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr); % pad with zeros, if this is the last batch
    end

    % apply high/low pass filtering
    % next four lines should be equivalent to filtfilt (which cannot be used because it requires float64)
    datr = filter(b1, a1, datr); % causal forward filter
    datr = flipud(datr); % reverse time
    datr = filter(b1, a1, datr); % causal forward filter again
    datr = flipud(datr); % reverse time back

    %datr    = gpufilter(buff, ops, chanMap); % apply filters and median subtraction

    % % --- end of gpufilter.m operations ---%
    % % ---------------------------------------------------------------------------------------------------------------
    
    if 0
        % % % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! % % %
        % Curious approach to soften potential abrupt steps between batches (??...due to demeaning and/or filtering?)
        % - ...not really keen on manipulating the source data like this --TBC
        datr(ntb + [1:ntb], :) = w_edge .* datr(ntb+[1:ntb], :)  +  (1-w_edge) .* datr_prev;
        
        datr_prev = datr(ntb +NT + [1:ops.ntbuff], :); % preserve trailing ntbuff samples to use for blending next batch
        % b/c NTbatch was set to NT+3*ntbuff, this [datr_prev] section
        % is actually a copy of the first few samples OF THE NEXT BATCH!
        % - so the w_edge transitioning is actually blending data with copy of itself that was demeaned/filtered with
        %   the preceeding batch
        % % % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! % % %
    end
    
    datr    = datr(ntb + (1:NT),:); % remove timepoints used as buffers
   
    datr    = datr * Wrot; % whiten the data and scale by [ops.scaleproc] for int16 range

    % datcpu  = gather(int16(datr')); % convert to int16, and gather on the CPU side
    % doesn't actually get sent to gpu right now (TBD: test if faster)
    count = fwrite(fidW, int16(datr'), 'int16'); % write this batch to binary file
    
    pb.check(ibatch) % update progress bar in command window
    
    if count~=numel(datr)
        error('Error writing batch %g to %s. Check available disk space.', ibatch, ops.fproc);
    end
end

% close the files
fclose(fidW); 
fclose(fid);

if getOr(ops, 'useMemMapping',1)
    % memory map [ops.fproc] file
    % - don't use Offset pv-pair, precludes using samples before tstart as buffer
    % - BUT challenge of reading data from fproc with tstart>0 remains
    %   - unless we preprocess ALL of raw file, including buffers in preprocessed data would create risky
    %     ambiguity between raw & processed time...
    %   - committing this first, then attempt 'least bad' alternative to preprocess from t0 to tend+ntbuff, regardless of tstart
    filename    = ops.fproc;
    datatype    = 'int16';
    chInFile    = ops.Nchan;
    bytes       = get_file_size(filename); % size in bytes of [new] preprocessed data file
    nSamp       = floor(bytes/chInFile/2);
    % check that file size is consistent with expected contents
    % NOTE: this will be slightly larger than original file (ops.sampsToRead)
    %       b/c rounded up to integer number of batches
    if nSamp ~= (ops.Nbatch*NT)
        keyboard
    end
    % memory map file
    rez.ops.fprocmmf         = memmapfile(filename, 'Format',{datatype, [chInFile nSamp], 'chXsamp'});
    fprintf('\tMemMapped preprocessed dat file:  %s\n\tas:  rez.ops.fprocmmf.Data.chXsamp\n', ops.fproc);
end

rez.Wrot    = gather(Wrot); % gather the whitening matrix as a CPU variable

cmdLog(sprintf('Finished preprocessing %d batches.', Nbatch), toc);

% rez.temp.Nbatch = Nbatch;  %?!?!!! why create rez.temp here?  Excised here & in subsequent processing stages

end %main function
