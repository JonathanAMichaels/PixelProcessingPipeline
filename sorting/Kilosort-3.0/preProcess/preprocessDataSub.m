function rez = preprocessDataSub(ops)
% this function takes an ops struct, which contains all the Kilosort2 settings and file paths
% and creates a new binary file of preprocessed data, logging new variables into rez.
% The following steps are applied:
% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

tic;
ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned

NT       = ops.NT ; % number of timepoints per batch
NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc

bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); %floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /NT); % number of data batches
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault); % if NchanTOT was left empty, then overwrite with the default

ops.igood = true(size(chanMap));

ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;


NTbuff      = NT + 3*ops.ntbuff; % we need buffers on both sides for filtering

rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;
