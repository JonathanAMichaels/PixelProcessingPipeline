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
% 2021-05-05  TBC  Updated to generate preprocessed data file ranging from t0:tend+ntbuff
%                  - accomodates non-zero tstart w/o sacrificing temporal correl w/ raw data file
% 2021-06-24  TBC  if [ops.CAR] value >1, will use as sliding window for median outlier (spike)
%                  when computing median (prevents high responsivity from skewing adjacent channels)
% 2023-09-07  SMO  Added option to remove channel delays, saving channel delays, and handle dummy
%                  channels in channel delay and whitening matrix calculations

%% Parse ops input & setup defaults
% record date & time of Kilosort execution
% - used by addFigInfo.m
ops.datenumSorted = now;

% tic for this function
t00 = tic;

% track git repo(s) with new utility (see <kilosortBasePath>/utils/gitStatus.m)
if getOr(ops, 'useGit', 1)
    ops = gitStatus(ops);
end

% show standard figs during processing (==2 for additional "debug" figure verbosity)
ops.fig = getOr(ops, 'fig', 1);
% Post-processing split/merge operations flags
%   1==template projections, 2==amplitudes, 0==don't split
ops.splitClustersBy = getOr(ops, 'splitClustersBy', 2);


ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned

NT          = ops.NT ; % number of timepoints per batch
NTbuff      = NT + 2*ops.ntbuff;
ops.NTbuff  = NTbuff;
%  NOTE: formerly "+ 3*ops.ntbuff" when linearly weighting current batch with previously processed copy of itself

NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc

bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /NT); % number of data batches
ops.Nbatch = Nbatch;

% ------------------------------------------------------------------------------------
% Preprocessed file size & samples
% ------------------------------------------------------------------------------------
ops.CAR = getOr(ops, 'CAR', 0); %% demean raw dat within channel with common *median* referencing

% still want processed batches to line up with tstart & tend in same way they do when tstart==0
% - sample based (not byte offset)
procBatchStarts = ops.tstart + (0:NT:NT*(ops.Nbatch-1)); % baseline batches of interest
if ops.tstart>0
    % append batches reaching back to start of file such that batch indices within tstart:tend are maintained
    % - as consequence, first batch may start with a negative value & contain <NT samples of **real data**
    % - thus, writing of that first batch will have to skip over any prepadded samples w/in typical non-buffer block of NT samples
    procBatchStarts = [fliplr( (ops.tstart-NT):-NT:(-NT+1)), procBatchStarts];
end
% - if all done correctly, get_batch.m should handle loading this appropriately by offsetting batch loading by tstart,
ops.NprocBatch = length(procBatchStarts);
ops.procBatchStarts = procBatchStarts;
% ------------------------------------------------------------------------------------

%% Load chanMap
[chanMap, xc, yc, kcoords, NchanMapTOT, numDummy] = loadChanMap(ops.chanMap); % function to load channel map file
%chanMap = ops.chanMap.chanMap; xc = ops.chanMap.xcoords;
%yc = ops.chanMap.ycoords; kcoords = ops.chanMap.kcoords;
%NchanMapTOT = length(chanMap);
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanMapTOT); % if .NchanTOT was left empty, then overwrite with n channels in file

ops.igood = true(size(chanMap));
ops.numDummy = numDummy;
ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops.chanMap = chanMap;

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;
rez.ops.numDummy = numDummy;

doCAR = ops.CAR; % shorthand

% set up the parameters of the filter % just one instance of filters
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass'); % butterworth filter with only 3 nodes (otherwise it's unstable for float32)
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high'); % the default is to only do high-pass filtering at 150Hz
end

cmdLog('Computing whitening matrix...', toc(t00));

% Compute whitening matrix
% this requires removing bad channels first
% broken and dummy channels are handled in myomatrix_binary.m, result is stored in chanMapAdjusted.mat
Wrot = get_whitening_matrix_faster(rez); % outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance of the data
if numDummy>0
    WrotDummy = eye(NchanTOT, NchanTOT);
    WrotDummy(1:NchanTOT-numDummy, 1:NchanTOT-numDummy) = Wrot;
    Wrot = WrotDummy;
end

disp('Whitening matrix computed...')
disp(Wrot)

cmdLog('Loading raw data and applying filters...', toc(t00));
if true
% open for reading raw data
fid = fopen(ops.fbinary, 'r');
if fid<3
    error('Could not open %s for reading.',ops.fbinary);
end
% open for writing processed data
fidW = fopen(ops.fproc,   'wb+');
if fidW<3
    error('Could not open %s for writing.',ops.fproc);    
end

ntb = ops.ntbuff;

% exponential smoothing
dnom = 3; % rate of padding exponential decay

allBatches = 1:ops.NprocBatch;

% Progress bar in command window
% pb = progBar(allBatches, 20);

for ibatch = allBatches
    % we'll create a preprocessed binary file by reading from the raw file (NOT memory mapped) batches of NT samples,
    % with each batch padded by ops.ntbuff samples from before & after, to have as buffers for filtering
    bstart = procBatchStarts(ibatch) - ntb; % start reading from ntbuffer samples before first batch start
    bsamp = bstart + (0:NTbuff-1); % determine sample indices (0-based)
    bsampTrue   = bsamp>=0 & bsamp<=nTimepoints; % determine number & validity of samples being read (if all are valid, this will sum to NTbuff)
    bsampTrueNT = bsampTrue(ntb+(1:NT)); % validity of batch samples (excluding buffers)
    
    offset = max(0, bstart*NchanTOT*2); % number of BYTES to offset start of standard read operation
    fseek(fid, offset, 'bof'); % fseek to batch start in raw file
    dat = fread(fid, [NchanTOT sum(bsampTrue)], '*int16'); % read and reshape. Assumes int16 data (which should perhaps change to an option)
    
    if isempty(dat)
        break; % this shouldn't really happen, unless we counted data batches wrong
    else
        nsampcurr = size(dat,2); % how many time samples the current batch has
    end
    
    % % ---------------------------------------------------------------------------------------------------------------
    % % --- step inside gpufilter.m operations ---%
    % - unpacked this utility function b/c unhappy with buffer padding before demeaning 
        
    % subsample only good channels & transpose for filtering
    datr = double(dat(chanMap,:))';    % dat dims now: [samples, channel]

    % --- Demean before padding ---
    % subtract within-channel means from each channel
    datr = datr - mean(datr, 1);  % nans not possible, since just converted from int16 raw dat values
    
        

    % CAR, common average referencing by median
    % -----------------------------
    % Ugly SUM(...,'omitnan') workaround for demeaning & subtracting in presence of nan without injecting spurrious zeros
    % -----------------------------
    % - "mean(...'omitnan')" is marginally faster than "nanmean(...)"
    % - BUT:  "nanmedian(...)" is significantly faster than "median(...'omitnan')"

    if doCAR
        if doCAR>1
            % Demean across cahnnels, exclude outlier values (spikes) from mean calc
            % - useful for moderate channel counts where reasonable for significant spiking
            %   activity to influence median value across channels
            datr = sum(cat(3, datr, repmat(-nanmedian(filloutliers(datr, nan), 2), [1,NchanMapTOT])), 3,'omitnan');
        else
            datr = sum(cat(3, datr, repmat(-nanmedian(datr, 2), [1,NchanMapTOT])), 3,'omitnan'); % subtract median across channels
        end
    end
    
    if any(isnan(datr))
        warning('NANs detected in batch raw data...inspect data validity and/or consider different CAR filtering option');
        keyboard
    end

    % Now can pad first & last batches with zeros
    %  if nsampcurr<NTbuff
    if bsampTrue(1) && bsampTrue(end)
        % all valid samples, do nothing
        
    elseif ~bsampTrue(1) && bsampTrue(end)
        % pad start of batch samples
        prepad = sum(~bsampTrue);
        fprintf(2, '\n\t~!~\tPre-padded batch %d with %d samples (%2.2f sec) ~!~\n', ibatch, prepad, prepad/ops.fs);
        if 0
            % prepend zeros if first batch
            datr = [zeros(prepad, size(datr,2)); datr];
        else
            % add smooth padding to start
            datr = [datr(1,:) .* exp(linspace(-prepad/dnom,0,prepad+1))'; datr(2:end, :)];
        end
        
    elseif bsampTrue(1) && ~bsampTrue(end) %ibatch~=1
        % pad end of batch samples
        postpad = sum(~bsampTrue);
        fprintf(2, '\n\t~!~\tPost-padded batch %d with %d samples (%2.2f sec) ~!~\n', ibatch, postpad, postpad/ops.fs);
        if 0
            % append zeros if end batch
            datr = [datr; zeros(postpad, size(datr,2))];
        else
            % add smooth padding to end
            datr = [datr(1:end-1,:); datr(end,:) .* exp(linspace(0,-postpad/dnom,postpad+1))'];
        end
        
    elseif ~bsampTrue(1) && ~bsampTrue(end)
        % This shouldn't happen, something is really wrong.
        keyboard;

    end
    
    
    % apply high/low pass filtering
    % next four lines should be equivalent to filtfilt (which cannot be used because it requires float64)
    datr = filter(b1, a1, datr); % causal forward filter
    datr = flipud(datr); % reverse time
    datr = filter(b1, a1, datr); % causal forward filter again
    datr = flipud(datr); % reverse time back

    % % --- end of gpufilter.m operations ---%
    % % ---------------------------------------------------------------------------------------------------------------
        
    datr    = datr(ntb + (1:NT),:); % remove timepoints used as buffers
   
    datr    = datr * Wrot; % whiten the data and scale by [ops.scaleproc] for int16 range

    % remove any batch samples that are not valid (can occur if tstart>0 and first raw batch sample is negative)
    datr(~bsampTrueNT,:) = [];
    
    % datcpu  = gather(int16(datr')); % convert to int16, and gather on the CPU side
    % doesn't actually get sent to gpu right now (TBD: test if faster)
    count = fwrite(fidW, int16(datr'), 'int16'); % write this batch to binary file
    
    %hit = pb.check(ibatch); % update progress bar in command window
    % updateProgressMessage(ibatch, ops.NprocBatch, t00,100,20);
    
    if count~=numel(datr)
        error('Error writing batch %g to %s. Check available disk space.', ibatch, ops.fproc);
    end
end
disp('Done.')

% close the files
fclose(fidW); 
fclose(fid);
end

if getOr(ops, 'useMemMapping',1)
    % memory map [ops.fproc] file
    % - don't use Offset pv-pair, precludes using samples before tstart as buffer
    filename    = ops.fproc;
    datatype    = 'int16';
    chInFile    = ops.Nchan;
    bytes       = get_file_size(filename); % size in bytes of [new] preprocessed data file
    nSamp       = floor(bytes/chInFile/2);
   % memory map file
    rez.ops.fprocmmf         = memmapfile(filename, 'Format',{datatype, [chInFile nSamp], 'chXsamp'});
    fprintf('\tMemMapped preprocessed dat file:  %s\n\tas:  rez.ops.fprocmmf.Data.chXsamp\n', ops.fproc);
end
script_dir = pwd; % get directory where repo exists
load(fullfile(script_dir, '/tmp/config.mat'));
if remove_channel_delays
    channelDelays = get_channel_delays(rez);
    rez.ops.channelDelays = channelDelays; % save channel delays to rez
    % figure(222); hold on;
    % remove channel delays from proc.dat by seeking through the batches
    % with ibatch*NT+max(channelDelays) and shifting each delayed channel backwards
    % by the appropriate amount found in channelDelays
    % this will effectively move some throwaway data to the end of all batches
    % but now the spikes will be aligned in time across channels
    fidOff = fopen(ops.fproc, 'r+');
    if fidOff < 3
        error('Could not open %s for reading.', ops.fbinary);
    end
    data = fread(fidOff, [NchanTOT inf], '*int16'); % read and reshape. Assumes int16 data
    % circularly shift each channel by the appropriate amount
    % plot(data')
    for i = 1:length(channelDelays)
        data(i, :) = circshift(data(i, :), channelDelays(i));
    end
    % plot(data' + max(abs(data(:)))) % plot shifted data
    fseek(fidOff, 0, 'bof'); % fseek to start in raw file, to overwrite
    fwrite(fidOff, data, 'int16');
    fclose(fidOff);
    disp('Removed channel delays from proc.dat, which were:')
    disp(channelDelays)
    % save(fullfile(myo_sorted_dir, 'channelDelays.mat'), 'channelDelays')
    disp('Delay information will be saved in ops.mat')
end
rez.Wrot    = gather(Wrot); % gather the whitening matrix as a CPU variable

cmdLog(sprintf('Finished preprocessing %d batches.', Nbatch), toc(t00));
rez.temp.Nbatch = Nbatch;
end %main function
