function [Wrot, CC] = get_whitening_matrix_faster(rez)
% based on a subset of the data, compute a channel whitening matrix
% this requires temporal filtering first (gpufilter)
% 
% [ks25] updates:
% - faster version uses memory mapped file reads & parallelized batch processing
%   to compute the whitening matrix
% - no longer uses gpufilter.m so that demeaning is consistently applied BEFORE padding incomplete batches
% 
% ---
% 202x-xx-xx  TBC  Evolved from standard get_whitening_matrix.m
% 2021-04-28  TBC  Cleaned & commented; useMemMapping = 1 by default
% 

ops = rez.ops;
Nbatch = ops.Nbatch;
twind = ops.twind; % Note: twind is pre-computed byte offset; twind~=tstart (for int16, twind==tstart*NchanTOT*2)
NchanTOT = ops.NchanTOT;
NT = ops.NT;
NTbuff = ops.NTbuff;
chanMap = ops.chanMap;
Nchan = rez.ops.Nchan;
xc = rez.xc;
yc = rez.yc;

% load data into patches, filter, compute covariance
fprintf('Getting channel whitening matrix... \n');

ibatch = 1;
ibatch = ibatch:ops.nSkipCov:Nbatch;

% scrappy progress bar in command window
pb = progBar(ibatch, 20);

if getOr(ops, 'useMemMapping', 1)
    %% Parallelize raw data loading (needs memmapfile implement for parallel read access)
        % slice ops for parallel
        ntbuff = ops.ntbuff; % NOTE: this is just the buffer length, not ".NTbuff" (which == the actual batch + buffer length)
        NT = ops.NT;
        bufferedNT = NT + 2*ntbuff;
        tend = ops.tend;
        tstart = ops.tstart;
        doCAR = logical(ops.CAR);
        CC = cell(1, numel(ibatch));
        [CC{:}] = deal(zeros(Nchan,  Nchan, 'single')); % we'll estimate the covariance from data batches, then add to this variable
    
        % set up the parameters of the filter % just one instance of filters
        if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
            [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass'); % butterworth filter with only 3 nodes (otherwise it's unstable for float32)
        else
            [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high'); % the default is to only do high-pass filtering at 150Hz
        end
        
        % Memory map the raw data file to m.Data.x of size [nChan, nSamples)
        %   mmfRaw = dir(ops.fbinary);
        bytesRaw    = get_file_size(ops.fbinary); % size in bytes of raw
        nsampRaw       = floor(bytesRaw/NchanTOT/2);
        mmfRaw      = memmapfile(ops.fbinary, 'Format',{'int16', [NchanTOT, nsampRaw], 'x'}); % ignore tstart here b/c we'll account for it on each mapped read (using .twind)
        %parmmf     = parallel.pool.Constant(mmf);
        tic
        parfor i = 1:length(ibatch)
        % for i = 1:length(ibatch)
            ii = ibatch(i);
            %offset = max(0, twind + 2*NchanTOT*((NT - ntbuff) * (ii-1) - 2*ntbuff));
            % index batch timepoints to read:  [ntbuff + NT + ntbuff]
            twindow = [max(1, tstart + NT*(ii-1) - ntbuff): min(nsampRaw, tstart + NT*(ii) + ntbuff)]; 

            % read from memory mapped file
            buff = mmfRaw.Data.x(:,twindow);
                        
            % % --- step inside gpufilter.m operations ---% 

            % subsample only good channels & transpose for filtering
            buff = single(buff(chanMap,:))';    % buff dims now: [time, channel]

            % --- Demean before padding ---
            % subtract within-channel means from each channel
            buff = buff - mean(buff, 1); % subtract mean of each channel

            % CAR, common average referencing by median
            if doCAR %getOr(ops, 'CAR', 1)
                buff = buff - median(buff, 2); % subtract median across channels
            end

            % Now can pad first & last batches with zeros
            nsampcurr = size(buff,1);
            if nsampcurr<bufferedNT
                if ii~=1
                    % append zeros if end batch
                    buff = [buff; zeros(bufferedNT-nsampcurr, size(buff,2))];
                else
                    % prepend zeros if first batch
                    buff = [zeros(bufferedNT-nsampcurr, size(buff,2)); buff];
                end
            end
            
            % apply high/low pass filtering
            % next four lines should be equivalent to filtfilt (which cannot be used because it requires float64)
            buff = filter(b1, a1, buff); % causal forward filter
            buff = flipud(buff); % reverse time
            buff = filter(b1, a1, buff); % causal forward filter again
            buff = flipud(buff); % reverse time back

            % % --- end of gpufilter.m operations ---% 
            
            buff = buff(ntbuff+1:end-ntbuff,:);   % remove timepoints used as buffers
            
            CC{i}        = (buff' * buff)/NT; % sample covariance
            %pb.check(i); % progress bar updates not meaningful when run on parallel pool
        end
        toc
        % scale & combine parallel outputs
        CC = cell2mat(cellfun(@(x) permute(x,[3,1,2]), CC', 'uni',0)); % shift dims of cell so that cell2mat maintains array dimensions
        CC = squeeze(mean(CC));
    %%
    
else
    % old single-thread method
    fid = fopen(ops.fbinary, 'r');
    tic
    CC = gpuArray.zeros( Nchan,  Nchan, 'single'); % we'll estimate the covariance from data batches, then add to this variable

    for i = 1:length(ibatch) %allBatches
        ii = ibatch(i);
        offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ii-1) - 2*ops.ntbuff));
        fseek(fid, offset, 'bof');
        buff = fread(fid, [NchanTOT NTbuff], '*int16');
        
        if isempty(buff)
            break;
        end
        nsampcurr = size(buff,2);
        if nsampcurr<NTbuff
            buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
        end
        
        datr    = gpufilter(buff, ops, chanMap); % apply filters and median subtraction
        
        CC        = CC + (datr' * datr)/NT; % sample covariance
        
        pb.check(ii) % update progress bar in command window
    end
    CC = CC / ceil((Nbatch-1)/ops.nSkipCov); % normalize by number of batches
    toc
    fclose(fid);
end

% CC = diag(diag(CC));
% mu = mean(diag(CC));

% CC = gpuArray(eye(size(CC), 'single'));
% CC = CC * mu;

% xp = cat(2, rez.xc, rez.yc);
% CC = mu * (.05 + .95 * kernel2D(xp, xp, 30));

% plot(diag(CC))

if ops.whiteningRange<Inf
    % if there are too many channels, a finite whiteningRange is more robust to noise in the estimation of the covariance
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather(CC), yc, xc, ops.whiteningRange); % this function performs the same matrix inversions as below, just on subsets of channels around each channel
else
    Wrot = whiteningFromCovariance(CC);
end
Wrot    = ops.scaleproc * Wrot; % scale this from unit variance to int 16 range. The default value of 200 should be fine in most (all?) situations.

fprintf('Channel-whitening matrix computed. \n');

end %main function
