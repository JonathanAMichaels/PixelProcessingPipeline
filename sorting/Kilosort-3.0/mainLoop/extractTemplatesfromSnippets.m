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
    [row, col] = isolated_peaks_buffered(-abs(dataRAW), ops);

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

dd_cpu = double(gather(dd));
[U, ~, ~] = svdecon(dd_cpu); % the PCs are just the left singular vectors of the waveforms
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

use_kmeans = true;
% initialize the template clustering 
if use_kmeans
    % project the waveforms onto the PCs
    % dd_pca = wPCA' * dd;
    % % compute k-means clustering of the waveforms
    % rng('default'); rng(1); % initializing random number generator for reproducibility
    % stream = RandStream('mlfg6331_64');  % Random number stream
    % options = statset('UseParallel', 1,'UseSubstreams', 1,'Streams', stream);
    % [cluster_id, ~, ~, Dist_from_K] = kmeans(dd_pca', nPCs, 'MaxIter', 1000, 'Replicates', 12, 'Display', 'final', 'Options', options);
    
    % % dd = dd(:,idx);
    % Dist_from_K_std = nan(nPCs,1);
    % wTEMP = gpuArray(nan(size(dd,1), nPCs));
    % N_spikes_in_cluster = nan(nPCs,1);
    % % dd_sorted_so_far = 0;
    % % for each cluster, find the standard deviation of the distances from center
    % % NOT IMPLEMENTED: for left bound, find the spike closest to the cluster center
    % % NOT IMPLEMENTED: for right bound, find the last spike within 1 std dev of the cluster center
    % % sort all spikes in dd first by cluster, then by distance from cluster center
    % [~, idx] = sort(cluster_id);
    % dd_tmp = gpuArray(nan(size(dd)));
    % wave_choice_left_bounds = ones(nPCs+1,1);
    % for K = 1:nPCs
    %     % count the number of spikes in each cluster
    %     N_spikes_in_cluster(K) = length(cluster_id(cluster_id==K));
    %     % % find the standard deviation of the distances from center for each cluster
    %     Dist_from_K_std(K) = std(Dist_from_K(cluster_id==K));
    %     % % within each cluster, sort spikes by distance from its cluster center
    %     idx_within_std_mask = cluster_id==K & Dist_from_K(:,K)<Dist_from_K_std(K);
    %     % [~, sorted_by_dist_idx_for_K] = sort(Dist_from_K(cluster_id==K), 'ascend');
    %     % choose top 10 closest spikes to the cluster center
    %     % top_best_10_spikes_for_K = dd(:, union(sorted_by_dist_idx_for_K(1:10), find(cluster_id==K)));
    %     % [~, sorted_idx_for_K] = sort(Dist_from_K(cluster_id==K & Dist_from_K < Dist_from_K_std(K)), 'ascend');

    %     % dd_for_K = dd(:,cluster_id==K);
    %     % sorted_dd_for_K = dd_for_K(:,sorted_idx_for_K);
    %     % dd(:,(dd_sorted_so_far+1):(dd_sorted_so_far+N_spikes_in_cluster(K))) = sorted_dd_for_K;
    %     % dd_sorted_so_far = dd_sorted_so_far + N_spikes_in_cluster(K);
        
    %     % compute minimum distance spike for each cluster
    %     % wTEMP(:,K) = mean(dd(:,idx_within_std_mask),2);
    %     wave_choice_left_bounds(K+1) = wave_choice_left_bounds(K)+sum(idx_within_std_mask)-1;
    %     dd_tmp(:,wave_choice_left_bounds(K):wave_choice_left_bounds(K+1)) = dd(:,idx(idx_within_std_mask));
    % end
    % % drop nan values
    % dd = dd_tmp(:,~isnan(dd_tmp(1,:)));
    dd_pca = wPCA' * dd;
    % compute k-means clustering of the waveforms
    rng('default'); rng(1); % initializing random number generator for reproducibility
    % stream = RandStream('mlfg6331_64');  % Random number stream
    % options = statset('UseParallel', 1,'UseSubstreams', 1,'Streams', stream);
    [cluster_id, ~, ~, Dist_from_K] = kmeans(dd_pca', nPCs, 'MaxIter', 10000, 'Replicates', 32, 'Display', 'final'); %, 'Options', options);
    % disp("replacing with K-means NOW")
    spikes = gpuArray(nan(size(dd)));
    number_of_spikes_to_use = nan(nPCs,1);
    for K=1:nPCs
        % use 90th percentile best examples from each cluster
        number_of_close_spikes = floor(sum(cluster_id==K)*0.9);
        [~,min_dist_idxs] = mink(Dist_from_K(:,K),number_of_close_spikes);
        spikes_to_use = dd(:,cluster_id==K & ismember(1:length(cluster_id),min_dist_idxs)');
        number_of_spikes_to_use(K) = size(spikes_to_use,2);
        % choose closest spikes to the cluster center
        spikes(:,sum(number_of_spikes_to_use(1:K-1))+1:sum(number_of_spikes_to_use,'omitnan')) = spikes_to_use;
    end
    % drop nan values
    spikes = spikes(:,~isnan(spikes(1,:)));
    % dbstop in extractTemplatesfromSnippets.m at 140
    total_num_spikes_used = sum(number_of_spikes_to_use);
    % take max of wave peaks 
    peaks = max(spikes, [], 1);
    wave_choice_left_bounds = [1; cumsum(number_of_spikes_to_use)];
    wave_choice_right_bounds = wave_choice_left_bounds(2:end);
    wave_choice_left_bounds = wave_choice_left_bounds(1:end-1);
    N_waves_between_choices = wave_choice_right_bounds - wave_choice_left_bounds;
    % define wave_choice_boundaries as all 1 std dev above and below the cluster centers
    % this avoids noisy spikes being chosen as templates
else
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
    total_num_spikes_used = length(idx);

    % assign non-uniform wave choice boundaries based on the amplutide of the peaks
    fraction_of_N_peaks = ceil(0.02*length(peaks));
    % get even distribution of spike amplitudes, treating positive and negative peaks separately
    % also skip the first and last chunks to avoid outliers
    num_max_peak_boundaries = ceil(length(max_peaks)/length(peaks)*nPCs);
    num_min_peak_boundaries = nPCs - num_max_peak_boundaries;
    if ~isempty(max_peaks) && ~isempty(min_peaks) % if there are both positive and negative peaks
        max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks,length(max_peaks))), max_peaks(end), num_max_peak_boundaries);
        min_peak_boundaries = linspace(min_peaks(1), min_peaks(end-min(fraction_of_N_peaks,length(min_peaks))), num_min_peak_boundaries);
        % combine the boundaries
        peak_boundaries = [max_peaks(1), max_peak_boundaries, min_peak_boundaries(2:end), min_peaks(end)];
    elseif ~isempty(max_peaks) % if there are only positive peaks
        max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks,length(max_peaks))), max_peaks(end), nPCs);
        peak_boundaries = [max_peaks(1), max_peak_boundaries, max_peaks(end)];
    elseif ~isempty(min_peaks) % if there are only negative peaks
        min_peak_boundaries = linspace(min_peaks(1), min_peaks(end-min(fraction_of_N_peaks,length(min_peaks))), nPCs);
        peak_boundaries = [min_peaks(1), min_peak_boundaries, min_peaks(end)];
    else
        error('No spikes found in the data!')
    end
    % find the closest peak to each boundary
    [~, wave_choice_boundaries] = min(abs(peaks-peak_boundaries'), [], 2);
    % N_waves_between_choices = diff(wave_choice_boundaries);

    % assign uniform wave choice boundaries
    % uniform_wave_choice_boundaries = round(linspace(1, length(peaks), nPCs+1));

    % wave_choice_left_bounds = wave_choice_boundaries(1:end-1);
    group_percent_expansion = 8; % percent
    group_size_expansion = ceil(group_percent_expansion/2/100*length(peaks));
    % define the boundaries, preventing negative values
    wave_choice_left_expand = max(wave_choice_boundaries-group_size_expansion, 1);
    wave_choice_left_bounds = wave_choice_left_expand(1:end-1);
    wave_choice_right_expand = min(wave_choice_boundaries+group_size_expansion, length(peaks));
    wave_choice_right_bounds = wave_choice_right_expand(2:end);
    N_waves_between_choices = wave_choice_right_bounds - wave_choice_left_bounds;
    disp(['Chunks overlap by ', num2str(group_percent_expansion), '%, which is ', num2str(group_size_expansion), ' spikes'])
    disp('Number of spikes in each chunk: ')
    disp(N_waves_between_choices')
end
figure(22);
plot((1:length(peaks))*ops.nt0, peaks, 'm'); hold on;
for iPeak = 1:length(peaks)
    if mod(iPeak,2)==0
        color = 'k';
        if use_kmeans
            plot((-ops.nt0min+1:ops.nt0min-1)+iPeak*ops.nt0, spikes(:,iPeak)', 'DisplayName', num2str(iPeak), 'Color', color)
        else
            plot((-ops.nt0min+1:ops.nt0min-1)+iPeak*ops.nt0, dd(:,idx(iPeak))', 'DisplayName', num2str(iPeak), 'Color', color)
        end
    end
end
title('peak amplitudes for each spike')

% plot vertical lines at the boundaries
for iBoundary = 1:length(wave_choice_left_bounds)
    % different color for each chunk, from magenta to cyan
    color = [1-iBoundary/length(wave_choice_left_bounds), iBoundary/length(wave_choice_left_bounds), 1];
    plot([1,1]*wave_choice_left_bounds(iBoundary)*ops.nt0, ylim, 'Color', color, 'LineWidth', 2)
    plot([1,1]*wave_choice_right_bounds(iBoundary)*ops.nt0, ylim, 'Color', color, 'LineWidth', 2)
end
if use_kmeans
    wTEMP = spikes(:,wave_choice_left_bounds); % initialize with a smooth range of amplitudes
else
    plot((1:length(peaks))*ops.nt0, peaks, 'm', 'LineWidth', 3);
    wTEMP = dd(:,idx(wave_choice_left_bounds)); % initialize with a smooth range of amplitudes
end
largest_CC_idx = 1;
N_tries_for_largest_CC_idx_so_far = 0;
best_CC_idxs = wave_choice_left_bounds;
sigma_time = 0.25; % ms of the gaussian kernel to focus penalty on central region of waveforms
sigma = ops.fs*sigma_time/1000; % samples
lowest_total_cost_for_each_chunk = 1e12*ones(1, length(wave_choice_left_bounds));
iter = 1;

while iter < 2
    % % find the pairs of waveforms that are too similar
    % [i,j] = find(CC>pos_cor_val | CC<neg_cor_val);
    % % remove i==j, which are always 1
    % i(i==j) = [];
    % % i_first = min(i);
    % % j_first = min(j);
    % % if max(i_first, j_first) > largest_CC_idx % make sure we do not go backwards in wave replacements
    % %     N_tries_for_largest_CC_idx_so_far = 0;
    % %     largest_CC_idx = max(i_first, j_first);
    % %     disp("Found a wave that meets the correlation threshold, using wave idx: " + num2str(best_CC_idxs(largest_CC_idx)))
    % %     sorted_CC = sort(CC(largest_CC_idx,:), 'descend');
    % %     disp(strcat("Total cross-channel correlation for this chunk was ", num2str(sum(abs(CC(largest_CC_idx,:))))))
    % %     disp(sorted_CC)
    % %     disp(CC)
    % % end
    % if N_tries_for_largest_CC_idx_so_far == 0
    %     disp(strcat("Now searching for initial template: ", num2str(largest_CC_idx)))
    % end
    % % start with the first pair, replace if second is too similar
    % if isempty(i)
    %     disp('no correlated pairs of templates found, continuing')
    %     correlated_pairs = false;
    %     disp(CC)
    % else
    % replace with next largest wave to check correlation, with each wave index relating to a amplitude-sorted chunk of the wave_choice_left_bounds
    if use_kmeans
        wTEMP(:,largest_CC_idx) = spikes(:,wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far);
    else
        wTEMP(:,largest_CC_idx) = dd(:,idx(wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far));
    end
    % multiply waveforms by a Gaussian with the sigma value
    % this is to make the correlation more sensitive to the central shape of the waveform
    wTEMP_for_corr = wTEMP .* gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma));
    CC = corr(wTEMP_for_corr);
    
    %% section to compute terms of the cost function
    % get residual of the waveform for this row of the CC matrix
    % sum the absolute value of the residual, scale by the absolute value of wTEMP_for_corr
    % this is to avoid using the waves with non-central shapes, by using it as a cost function
    wTEMP_gaussian_residual = sum(abs(wTEMP(:,largest_CC_idx) - wTEMP_for_corr(:,largest_CC_idx))) / sum(abs(wTEMP_for_corr(:,largest_CC_idx)));
    % compute the penalty for the highest single correlation in the CC matrix for this template choice
    if largest_CC_idx == 1
        highest_template_similarity_penalty = max( max(CC(largest_CC_idx,2:end), ...
                                                max(CC(2:end,largest_CC_idx))) );
    else
        highest_template_similarity_penalty = max( max(CC(1:largest_CC_idx-1,largest_CC_idx)), ...
                                                max(CC(largest_CC_idx,1:largest_CC_idx-1)));
    end
    highest_template_similarity_penalty = max(highest_template_similarity_penalty, 0); % make sure it is not negative
    % compute sum of all similarities with other template choices
    % if iter == 1 % handle different cases for first and second iterations
    if largest_CC_idx == 1
        corr_sum_with_other_template_choices = sum(max(CC(largest_CC_idx,2:end),0)) +...
                                                sum(max(CC(2:end,largest_CC_idx),0));
    else
        corr_sum_with_other_template_choices = sum(sum(max(CC(1:largest_CC_idx-1,largest_CC_idx),0))) +...
                                                    sum(sum(max(CC(largest_CC_idx,1:largest_CC_idx-1),0)));
    end
    total_cost_for_wave = corr_sum_with_other_template_choices + 12*nPCs*wTEMP_gaussian_residual + 12*nPCs*highest_template_similarity_penalty;
    % else
    % for second iteration, we should check all off-diagonal effects, but normalize by the
    % number of elements used, to allow comparable costs with iteration 1, when we only check the
    % off-diagonal elements of 1:largest_CC_idx rows and columns
    % if largest_CC_idx == 1
    %     corr_sum_with_other_template_choices = sum(sum(max(CC(~logical(eye(size(CC)))),0))) / (((nPCs-1)*(nPCs))/(2*nPCs-2));
    % else
    %     corr_sum_with_other_template_choices = sum(sum(max(CC(~logical(eye(size(CC)))),0))) / (((nPCs-1)*(nPCs))/(2*largest_CC_idx-2));
    %     % get the total cost for this chunk, by adding all the cross-channel correlations and the residual
    % end
    % total_cost_for_wave = corr_sum_with_other_template_choices + nPCs*(wTEMP_gaussian_residual + 2*highest_template_similarity_penalty);
    
    %% section to choose the best wave for this chunk
    if (total_cost_for_wave < lowest_total_cost_for_each_chunk(largest_CC_idx)) && ~ismember(wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far, best_CC_idxs)
        lowest_total_cost_for_each_chunk(largest_CC_idx) = total_cost_for_wave;
        best_CC_idxs(largest_CC_idx) = wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far;
    end
    N_tries_for_largest_CC_idx_so_far = N_tries_for_largest_CC_idx_so_far + 1;
    % terminate if we have tried all waves in the amplitude-sorted chunk
    if N_tries_for_largest_CC_idx_so_far >= N_waves_between_choices(largest_CC_idx) || N_tries_for_largest_CC_idx_so_far >= total_num_spikes_used
        % wrap disp lines to avoid going over 100 characters
        disp(strcat("Tried all waves in amplitude-sorted chunk ", num2str(largest_CC_idx), ", using wave idx with best CC: ", num2str(best_CC_idxs(largest_CC_idx))))
        % sorted_CC = sort(CC(largest_CC_idx,:), 'descend');
        disp(strcat("Total cross-channel correlation for this chunk ", num2str(sum(abs(CC(largest_CC_idx,:))))))
        disp("Residual cost for this chunk was " + num2str(wTEMP_gaussian_residual))
        disp("Highest template similarity penalty for this chunk was " + num2str(highest_template_similarity_penalty))
        disp("Final cost for this chunk was (including cumulative cost): " + num2str(lowest_total_cost_for_each_chunk(largest_CC_idx)))
        % disp(sorted_CC)
        disp(CC)
        largest_CC_idx = largest_CC_idx + 1;
        N_tries_for_largest_CC_idx_so_far = 0;
        if largest_CC_idx > nPCs
            largest_CC_idx = 1;
            % correlated_pairs = false; % terminate the while loop
            disp("Final waveforms chosen:")
            disp(best_CC_idxs)
            disp("Final CC matrix:")
            disp(CC)
            iter = iter + 1;
        end
    end
end

if ~use_kmeans
    wTEMP(:,1:nPCs) = dd(:,idx(best_CC_idxs(1:nPCs)));
else
    wTEMP(:,1:nPCs) = spikes(:,best_CC_idxs(1:nPCs));
end
% % multiply by a Gaussian with the sigma value
% wTEMP = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 1, size(wTEMP,2));
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % normalize them
if ops.fig == 1
    % wTEMP_for_CC_final = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 1, size(wTEMP,2));
    figure(2); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i*1);
        % plot standardized Gaussian multiplied waveforms for comparison
        plot(wTEMP(:,i)./sum(wTEMP(:,i).^2,1).^.5+i, 'r');
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

% dd_gaussian = dd .* repmat(gausswin(size(dd,1), (size(dd,1)-1)/(2*sigma)), 1, size(dd,2));
% randomly sample nPCs units
% wTEMP = dd(:, randperm(size(dd,2), nPCs));
% if ~use_kmeans

% dd_pca = wPCA' * dd;
% % compute k-means clustering of the waveforms
% rng('default'); rng(1); % initializing random number generator for reproducibility
% % stream = RandStream('mlfg6331_64');  % Random number stream
% % options = statset('UseParallel', 1,'UseSubstreams', 1,'Streams', stream);
% [cluster_id, ~, ~, Dist_from_K] = kmeans(dd_pca', nPCs, 'MaxIter', 10000, 'Replicates', 12, 'Display', 'final');%, 'Options', options);
% % disp("replacing with K-means NOW")
% spikes = gpuArray(nan(size(dd)));
% number_of_spikes_to_use = nan(nPCs,1);
% for K=1:nPCs
%     % use top quarter best examples from each cluster
%     number_of_close_spikes = round(sum(cluster_id==K)/4);
%     [~,min_dist_idxs] = mink(Dist_from_K(:,K),number_of_close_spikes);
%     spikes_to_use = dd(:,cluster_id==K & ismember(1:length(cluster_id),min_dist_idxs)');
%     number_of_spikes_to_use(K) = size(spikes_to_use,2);
%     % choose closest spikes to the cluster center
%     spikes(:,sum(number_of_spikes_to_use(1:K-1))+1:sum(number_of_spikes_to_use,'omitnan')) = spikes_to_use;
% end
% spikes = spikes(:,~isnan(spikes(1,:)));
if use_kmeans
    wTEMP = dd(:,randperm(size(dd,2), nPCs)); % removing this line will cause KS to not find spikes sometimes... ???
    wTEMP(:,1:6) = spikes(:,best_CC_idxs(1:6));
    spikes_gauss_windowed = spikes .* gausswin(size(spikes,1), (size(spikes,1)-1)/(2*sigma));
    wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % standardize the new clusters
    for i = 1:10
        % at each iteration, assign the waveform to its most correlated cluster
        CC = wTEMP' * spikes_gauss_windowed;
        [amax, imax] = max(CC,[],1); % find the best cluster for each waveform
        for j = 1:nPCs
            wTEMP(:,j)  = spikes(:,imax==j) * amax(imax==j)'; % weighted average to get new cluster means
            % if a template had no matches and therefore has NaN's,
            % use the mean of top 10th percentil in that k-means cluster instead
            if sum(abs(wTEMP(:,j)))==0 % make sure the template is not all zeros after the weighted average
                [~,min_dist_idxs] = mink(Dist_from_K(:,j),ceil(sum(cluster_id==j)/10));
                spikes_to_use = dd(:,cluster_id==j & ismember(1:length(cluster_id),min_dist_idxs)');
                wTEMP(:,j) = mean(spikes_to_use,2);
            end
        end
        wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % standardize the new clusters
    end
else
    for i = 1:10
        % at each iteration, assign the waveform to its most correlated cluster
        CC = wTEMP' * dd;
        [amax, imax] = max(CC,[],1); % find the best cluster for each waveform
        for j = 1:nPCs
            wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)'; % weighted average to get new cluster means
        end
        wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5; % standardize the new clusters
    end
end


if ops.fig == 1
    figure(3); hold on;
    for i = 1:nPCs
        plot(wTEMP(:,i)+i);
    end
    title('prototype templates');
end
% dbstop in extractTemplatesfromSnippets.m at 448
% disp('stop')
