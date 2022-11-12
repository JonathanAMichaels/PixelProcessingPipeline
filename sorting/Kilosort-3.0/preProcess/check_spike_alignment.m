function rez = check_spike_alignment(rez)

ops = rez.ops;

NT       = ops.NT ; % number of timepoints per batch
NchanTOT = ops.Nchan; % total number of channels in the raw binary file

bytes       = get_file_size(ops.fproc); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); %floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /NT); % number of data batches

NTbuff      = NT; % we need buffers on both sides for filtering

fid        = fopen(ops.fproc,   'r');
if fid<3
    error('Could not open %s for writing.',ops.fproc);
end


ntb = 0;

for ibatch = 1:Nbatch
    % we'll create a binary file of batches of NT samples, which overlap consecutively on ops.ntbuff samples
    % in addition to that, we'll read another ops.ntbuff samples from before and after, to have as buffers for filtering
    offset = max(0, ops.twind + 2*NchanTOT*(NT * (ibatch-1) - ntb)); % number of samples to start reading at.

    fseek(fid, offset, 'bof'); % fseek to batch start in raw file

    buff = fread(fid, [NchanTOT NTbuff], '*int16'); % read and reshape. Assumes int16 data (which should perhaps change to an option)

    if isempty(buff)
        break; % this shouldn't really happen, unless we counted data batches wrong
    end
    nsampcurr = size(buff,2); % how many time samples the current batch has
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr); % pad with zeros, if this is the last batch
    end
    if offset==0
        bpad = repmat(buff(:,1), 1, ntb);
        buff = cat(2, bpad, buff(:, 1:NTbuff-ntb)); % The very first batch has no pre-buffer, and has to be treated separately
    end



end

fclose(fid);

