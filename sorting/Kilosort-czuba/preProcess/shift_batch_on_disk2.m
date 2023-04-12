function [dprev, dat_cpu, dat, shifts] = shift_batch_on_disk2(rez, ibatch, shifts, ysamp, sig, fdest)
% register one batch of a whitened binary file
%
% This is only filtering on channels, not timepoints, so why is it applying any edge blending with
% previous batch [dprev] at all???  --TBC
% 
% [ks25] updates:
% - Excised usage of [rez.temp.Nbatch]
% - No more [dprev] input or output, now just fdest for destination file handle (open ONCE outside of this fxn)
% 

ops = rez.ops;
Nbatch      = rez.ops.Nbatch;
NT  	      = ops.NT;

% batches start at these sample timepoints
if isfield(ops, 'procBatchStarts')
    % use correct batch starts when preprocessing is done from "true t=0" and/or ops.trange ~= [0 inf];
    % - allows correct batch & buffer handling w/ get_batch.m (**ks25 rewrite**)
    % - facilitates waveform recall from timestamps w/o need for magic number offset in downstream analysis stages
    batchstart = rez.ops.procBatchStarts( rez.ops.procBatchStarts >= rez.ops.tstart );
    % error check
    assert(length(batchstart)==Nbatch, ['Error while applying batch shifts to preprocessed file: \n\t%s\n',...
        '~!~\tExpected %d batches, but resolved %d batch starts from [rez.ops] struct\n'], rez.ops.fproc, Nbatch, length(batchstart));
else
    batchstart = 0:NT:NT*Nbatch;
end

offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes

% upsample the shift for each channel using interpolation
if length(ysamp)>1
    shifts = interp1(ysamp(1:length(shifts)), shifts, rez.yc, 'makima', 'extrap');
end

% load the batch w/o buffer & w/o scaling (will be writing directly back to int16 data anyway)
dat = get_batch(ops, ibatch, [], 'nobuffer','noscale');
% fclose all;
% ntb = ops.ntbuff;
% fid = fopen(ops.fproc, 'r+');
% fseek(fid, offset, 'bof');
% dat = fread(fid, [ops.Nchan NT+ntb], '*int16')';

% 2D coordinates for interpolation 
xp = cat(2, rez.xc, rez.yc);

% 2D kernel of the original channel positions 
Kxx = kernel2D(xp, xp, sig);

% 2D kernel of the new channel positions
yp = xp;
yp(:, 2) = yp(:, 2) - shifts; % * sig;
Kyx = kernel2D(yp, xp, sig);

% kernel prediction matrix
M = Kyx /(Kxx + .01 * eye(size(Kxx,1)));

% the multiplication has to be done on the GPU      % dati = gpuArray(single(dat)) * gpuArray(M)';
% - dat is already on GPU when read with get_batch.m
dati = dat * gpuArray(M)';

% don't do this, not necessary to mess with the buffer here
% w_edge = linspace(0, 1, ntb)';
% dati(1:ntb, :) = w_edge .* dati(1:ntb, :) + (1 - w_edge) .* dprev;

% if size(dati,1)==NT+ntb
%     dprev = dati(NT+[1:ntb], :);
% else
%     dprev = [];
% end
% dati = dati(1:NT, :);

dat_cpu = gather(int16(dati));


% we want to write the aligned data back to the same file
fseek(fdest, offset, 'bof');
fwrite(fdest, dat_cpu', 'int16'); % write this batch to binary file

% fclose(fid);

