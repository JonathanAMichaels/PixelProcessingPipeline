function [dprev, dat_cpu, dat, shifts] = ...
    shift_batch_on_disk2(rez, ibatch, shifts, sig, dprev)
% register one batch of a whitened binary file

ops = rez.ops;

% switcheroo
origNchan = ops.Nchan;
ops.Nchan = ops.NchanTOT;

Nbatch      = rez.temp.Nbatch;
NT  	      = ops.NT;

batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes


% load the batch
fclose all;
ntb = ops.ntbuff;
fidR = fopen(ops.fbinary, 'r');
fseek(fidR, offset, 'bof');
dat = fread(fidR, [ops.Nchan NT+ntb], '*int16')';
fclose(fidR);

% trim back down
ops.Nchan = origNchan;
dat = dat(:,1:ops.Nchan);

% 2D coordinates for interpolation 
xp = cat(2, rez.xc, rez.yc);

% 2D kernel of the original channel positions 
Kxx = kernel2D(xp, xp, sig);
% 2D kernel of the new channel positions
yp = xp;
yp(:, 2) = yp(:, 2) - shifts'; % * sig;
Kyx = kernel2D(yp, xp, sig);

% kernel prediction matrix
M = Kyx /(Kxx + .01 * eye(size(Kxx,1)));

% the multiplication has to be done on the GPU
dati = gpuArray(single(dat)) * gpuArray(M)';

w_edge = linspace(0, 1, ntb)';
dati(1:ntb, :) = w_edge .* dati(1:ntb, :) + (1 - w_edge) .* dprev;

if size(dati,1)==NT+ntb
    dprev = dati(NT+[1:ntb], :);
else
    dprev = [];
end
if size(dati,1) >= NT
    dati = dati(1:NT, :);
end


dat_cpu = gather(int16(dati));


fidW = fopen(ops.fproc, 'r+');
offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes
fseek(fidW, offset, 'bof');
fwrite(fidW, dat_cpu', 'int16'); % write this batch to binary file

fclose(fidW);

