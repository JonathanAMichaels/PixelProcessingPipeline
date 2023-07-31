function [wTEMP, wPCA] = extractTemplatesfromSnippets(rez, nPCs)
    % this function is very similar to extractPCfromSnippets.
    % outputs not just the PC waveforms, but also the template "prototype", 
    % basically k-means clustering of 1D waveforms. 
    
ops = rez.ops;

% skip every this many batches
nskip = getOr(ops, 'nskip', 25);

Nbatch      = rez.temp.Nbatch;
NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

fid = fopen(ops.fproc, 'r'); % open the preprocessed data file

k = 0;
dd = gpuArray.zeros(ops.nt0, 5e4, 'single'); % preallocate matrix to hold 1D spike snippets
if ops.fig == 1
    figure(1); hold on;
end
for ibatch = 1:nskip:Nbatch
    offset = 2 * ops.Nchan*batchstart(ibatch);
    fseek(fid, offset, 'bof');
    dat = fread(fid, [ops.Nchan NT], '*int16');
    dat = dat';

    % move data to GPU and scale it back to unit variance
    dataRAW = gpuArray(dat);
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;


    % find isolated spikes from each batch
    [row, col, mu] = isolated_peaks_new(-abs(dataRAW), ops);

    % for each peak, get the voltage snippet from that channel
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);
    c = sq(clips(:, :));
    if ops.fig == 1
        plot(c)
    end
    if k+size(c,2)>size(dd,2)
        dd(:, 2*size(dd,2)) = 0;
    end
    
    dd(:, k + [1:size(c,2)]) = c;
    k = k + size(c,2);
    if k>1e5
        break;
    end
end
fclose(fid);
if ops.fig == 1
    title('local isolated spikes (1D voltage waveforms)');
end
% discard empty samples
dd = dd(:, 1:k);

% initialize the template clustering with random waveforms
% wTEMP = dd(:, randperm(size(dd,2), nPCs));
% sort dd by largest peak amplitude, with positive peaks first
% if negative peaks are larger, those spikes will be added later in the array
% check if each spikes max or min is larger
% if the max is larger, use the max_idx, otherwise use the min_idx
% this is to make sure that the spikes are segregated by polarity
% and that the largest spikes are used first
maxes_for_each_spike = max(dd);
mins_for_each_spike = min(dd);
max_larger_mask = maxes_for_each_spike > abs(mins_for_each_spike);
[max_peaks, max_larger_sorted_idx] = sort(max(dd(:,max_larger_mask)), 'descend');
[min_peaks, min_larger_sorted_idx] = sort(min(dd(:,~max_larger_mask)), 'descend');
if ~isempty(max_peaks) && ~isempty(min_peaks)
    peaks = [max_peaks, min_peaks];
elseif ~isempty(max_peaks)
    peaks = max_peaks;
elseif ~isempty(min_peaks)
    peaks = min_peaks;
else
    error('No spikes found in the data!')
end
max_mask_idx = find(max_larger_mask);
min_mask_idx = find(~max_larger_mask);
idx = [max_mask_idx(max_larger_sorted_idx), min_mask_idx(min_larger_sorted_idx)];
figure(22)
plot((1:length(peaks))*ops.nt0, peaks, 'm'); hold on;
for iPeak = 1:length(peaks)
    if mod(iPeak,2)==0
        color = 'k';
        plot((-ops.nt0min+1:ops.nt0min-1)+iPeak*ops.nt0, dd(:,idx(iPeak))', 'DisplayName', num2str(iPeak), 'Color', color)
    end
end
title('peak amplitudes for each spike')
do_uniform_wave_choice = false;
if do_uniform_wave_choice
    % assign uniform wave choice boundaries
    wave_choice_boundaries = round(linspace(1, length(peaks), nPCs+1));
    N_waves_between_choices = diff(wave_choice_boundaries);
else
    % assign non-uniform wave choice boundaries based on the amplutide of the peaks
    fraction_of_N_peaks = ceil(0.02*length(peaks));
    % get even distribution of spike amplitudes, treating positive and negative peaks separately
    % also skip the first and last chunks to avoid outliers
    num_max_peak_boundaries = ceil(length(max_peaks)/length(peaks)*nPCs);
    num_min_peak_boundaries = nPCs - num_max_peak_boundaries;
    if ~isempty(max_peaks) && ~isempty(min_peaks)
        max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks,length(max_peaks))), max_peaks(end), num_max_peak_boundaries);
        min_peak_boundaries = linspace(min_peaks(1), min_peaks(end-min(fraction_of_N_peaks,length(min_peaks))), num_min_peak_boundaries);
        % combine the boundaries
        peak_boundaries = [max_peaks(1), max_peak_boundaries, min_peak_boundaries(2:end), min_peaks(end)];
    elseif ~isempty(max_peaks)
        max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks,length(max_peaks))), max_peaks(end), nPCs);
        peak_boundaries = [max_peaks(1), max_peak_boundaries, max_peaks(end)];
    elseif ~isempty(min_peaks)
        min_peak_boundaries = linspace(min_peaks(1), min_peaks(end-min(fraction_of_N_peaks,length(min_peaks))), nPCs);
        peak_boundaries = [min_peaks(1), min_peak_boundaries, min_peaks(end)];
    else
        error('No spikes found in the data!')
    end
    % find the closest peak to each boundary
    [~, wave_choice_boundaries] = min(abs(peaks-peak_boundaries'), [], 2);
    N_waves_between_choices = diff(wave_choice_boundaries);
end
% plot vertical lines at the boundaries
for iBoundary = 1:length(wave_choice_boundaries)
    plot([1,1]*wave_choice_boundaries(iBoundary)*ops.nt0, ylim, 'm--')
end
plot((1:length(peaks))*ops.nt0, peaks, 'm', 'LineWidth', 3);
wave_choice_left_bound = wave_choice_boundaries(1:end-1);
wTEMP = dd(:,idx(wave_choice_left_bound)); % got a smooth range of amplitudes
correlated_pairs = true;
% compute the pairwise correlation of each timeseries 
CC = corr(wTEMP);
% check pairwise correlations between the random waveforms and
% replace any columns that are too similar until they are all unique
% do not check the diagonal, which is always 1
% initialize variables
pos_cor_val = 0.5;
neg_cor_val = -0.5;
largest_CC_idx = 1;
N_tries_for_largest_CC_idx_so_far = 0;
best_CC_idxs = wave_choice_left_bound;
sigma_time = 0.25; % ms
sigma = ops.fs*sigma_time/1000; % samples
lowest_total_cost_for_subsection = 1e12;
while correlated_pairs==true
    [i,j] = find(CC>pos_cor_val | CC<neg_cor_val);
    % remove i==j
    i(i==j) = [];
    % i_first = min(i);
    % j_first = min(j);
    % if max(i_first, j_first) > largest_CC_idx % make sure we do not go backwards in wave replacements
    %     N_tries_for_largest_CC_idx_so_far = 0;
    %     largest_CC_idx = max(i_first, j_first);
    %     disp("Found a wave that meets the correlation threshold, using wave idx: " + num2str(best_CC_idxs(largest_CC_idx)))
    %     sorted_CC = sort(CC(largest_CC_idx,:), 'descend');
    %     disp(strcat("Total cross-channel correlation for this subsection was ", num2str(sum(abs(CC(largest_CC_idx,:))))))
    %     disp(sorted_CC)
    %     disp(CC)
    % end
    if N_tries_for_largest_CC_idx_so_far == 0
        disp(strcat("Now searching for initial template: ", num2str(largest_CC_idx)))
    end
    % start with the first pair, replace if second is too similar
    if isempty(i)
        disp('no correlated pairs of templates found, continuing')
        correlated_pairs = false;
        disp(CC)
    else
        % disp('correlated pairs of templates found, replacing')
        % replace with next largest wave to check correlation, with each wave index relating to a amplitude-sorted subsection of the wave_choice_left_bound
        wTEMP(:,largest_CC_idx) = dd(:,idx(wave_choice_left_bound(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far));
        % multiply waveforms by a Gaussian with the sigma value
        % this is to make the correlation more sensitive to the central shape of the waveform
        wTEMP_for_corr = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 1, size(wTEMP,2));
        CC = corr(wTEMP_for_corr);
        % get residual of the waveform for this row of the CC matrix
        % sum the absolute value of the residual, scale by the absolute value of wTEMP_for_corr
        % this is to avoid using the waves with non-central shapes, by using it as a cost function
        wTEMP_gaussian_residual = sum(abs(wTEMP(:,largest_CC_idx) - wTEMP_for_corr(:,largest_CC_idx))) / sum(abs(wTEMP_for_corr(:,largest_CC_idx)));
        total_cost_for_wave = sum(sum(abs(CC(1:largest_CC_idx,:)))) + nPCs*wTEMP_gaussian_residual;
        if total_cost_for_wave < lowest_total_cost_for_subsection
            lowest_total_cost_for_subsection = total_cost_for_wave;
            best_CC_idxs(largest_CC_idx) = wave_choice_left_bound(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far;
        end
        N_tries_for_largest_CC_idx_so_far = N_tries_for_largest_CC_idx_so_far + 1;
        % terminate if we have tried all waves in the amplitude-sorted subsection
        if N_tries_for_largest_CC_idx_so_far >= N_waves_between_choices(largest_CC_idx)
            % wrap disp lines to avoid going over 100 characters
            disp(strcat("Tried all waves in amplitude-sorted subsection ", num2str(largest_CC_idx), ", using wave idx with best CC: ", num2str(best_CC_idxs(largest_CC_idx))))
            % sorted_CC = sort(CC(largest_CC_idx,:), 'descend');
            disp(strcat("Total cross-channel correlation for this subsection ", num2str(sum(abs(CC(largest_CC_idx,:))))))
            disp("Residual cost for this subsection was " + num2str(wTEMP_gaussian_residual))
            disp("Final cost for this subsection was (including cumulative cost)" + num2str(lowest_total_cost_for_subsection))
            % disp(sorted_CC)
            disp(CC)
            wTEMP(:,largest_CC_idx) = dd(:,idx(best_CC_idxs(largest_CC_idx)));
            largest_CC_idx = largest_CC_idx + 1;
            N_tries_for_largest_CC_idx_so_far = 0;
            lowest_total_cost_for_subsection = 1e12;
            if largest_CC_idx > nPCs
                correlated_pairs = false;
                disp("Final waveforms chosen:")
                disp(best_CC_idxs)
                disp("Final CC matrix:")
                disp(CC)
            end
        end
    end
end
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % normalize them
if ops.fig == 1
    wTEMP_for_CC_final = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 1, size(wTEMP,2));
    figure(2); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i*1);
        % plot standardized Gaussian multiplied waveforms for comparison
        plot(wTEMP_for_CC_final(:,i)./sum(wTEMP_for_CC_final(:,i).^2,1).^.5+i, 'r');
        % plot the gaussian
        plot(i+gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 'k');
    end
    title('initial templates');
end
% % take harmonic mean of all highly correlated waveforms
% CC = wTEMP' * dd;
% [amax, imax] = max(CC,[],1); % find the best cluster for each waveform
% for j = 1:nPCs
%     wTEMP(:,j) = harmmean(dd(:,(imax==j & amax>0.5)),2); % weighted average to get new cluster means
% end

for i = 1:10
  % at each iteration, assign the waveform to its most correlated cluster
   CC = wTEMP' * dd;
   [amax, imax] = max(CC,[],1); % find the best cluster for each waveform
   for j = 1:nPCs
      wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)'; % weighted average to get new cluster means
   end
   wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % standardize the new clusters
end
% multiply final waveforms by a Gaussian with the sigma value times 2
% wTEMP = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*2*sigma)), 1, size(wTEMP,2));


if ops.fig == 1
    figure(3); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i);
    end
    title('prototype templates');
end

dd = double(gather(dd));
[U Sv V] = svdecon(dd); % the PCs are just the left singular vectors of the waveforms
% if ops.fig == 1
%     figure(4); hold on;
%     for i = 1:nPCs
%         plot(U(:,i)+i*1);
%     end
%     title(strcat("Top ", num2str(nPCs), " PCs"));
% end
wPCA = gpuArray(single(U(:, 1:nPCs))); % take as many as needed
% adjust the arbitrary sign of the first PC so its peak is downward
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(ops.nt0min,1));
