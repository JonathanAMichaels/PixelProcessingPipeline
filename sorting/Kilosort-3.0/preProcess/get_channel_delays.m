function [channel_delays] = get_channel_delays(rez)
% based on a subset of the data, compute a channel whitening matrix
% this requires temporal filtering first (gpufilter)

ops = rez.ops;
Nbatch = ops.Nbatch;
twind = ops.twind;
NchanTOT = ops.NchanTOT;
NT = ops.NT;
NTbuff = ops.NTbuff;
Nchan = rez.ops.Nchan;

fprintf('Getting channel delays... \n');
fid = fopen(ops.fbinary, 'r');
maxlag = ops.fs/500; % 2 ms lag

% we'll estimate the autocorrelation of each channel from data batches
ibatch = 1;
chan_CC = zeros(2*maxlag+1, NchanTOT^2, 'single', 'gpuArray');
while ibatch<=Nbatch
    offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');

    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    buff = gpuArray(buff);
    chan_CC = chan_CC + xcorr(abs(buff'), maxlag, 'coeff');

    ibatch = ibatch + ops.nSkipCov; % skip this many batches
end
% normalize result by number of batches
chan_CC = chan_CC / ceil((Nbatch-1) / ops.nSkipCov);

fclose(fid);

% find the channel which is earliest in time, relative to other channels
% last_delays = 2*maxlag*ones(1,Nchan)+1;
last_maxes = zeros(1,Nchan);
[chan_peak_maxes, chan_peak_locs] = max(chan_CC, [], 1);
for iChan = 1:Nchan
    these_maxes = chan_peak_maxes(Nchan*(iChan-1)+1:Nchan*iChan);
    this_chan_peak_locs = chan_peak_locs(Nchan*(iChan-1)+1:Nchan*iChan);
    % these_delays = this_chan_peak_locs - maxlag - 1;
    if sum(these_maxes) > sum(last_maxes) % if these delays produce higher correlation
        best_peak_locs = this_chan_peak_locs;
        last_maxes = these_maxes;
    end
end
% use the earliest channel as a reference to compute delays
channel_delays = gather(best_peak_locs - maxlag - 1); % -1 because of zero-lag
disp("Channel delays computed: ")
disp(reshape(chan_peak_locs, Nchan, Nchan)-maxlag-1)
disp("Correlation values trying each reference channel: ")
disp(reshape(chan_peak_maxes, Nchan, Nchan))
disp(" + ___________________________________________________________")
disp(sum(reshape(chan_peak_maxes, Nchan, Nchan)))
disp("Using best reference channel, with maximal correlation: ")
disp(sum(last_maxes))
disp("Channel delays using best reference channel: ")
disp(channel_delays)
end

