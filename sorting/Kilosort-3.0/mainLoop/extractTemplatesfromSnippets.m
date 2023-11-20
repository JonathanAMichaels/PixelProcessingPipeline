function [wTEMP, wPCA] = extractTemplatesfromSnippets(rez, nPCs)
    % this function is very similar to extractPCfromSnippets.
    % outputs not just the PC waveforms, but also the template "prototype",
    % basically k-means clustering of 1D waveforms.

    ops = rez.ops;

    % skip every this many batches
    nskip = getOr(ops, 'nskip', 25);

    Nbatch = rez.temp.Nbatch;
    NT = ops.NT;
    batchstart = 0:NT:NT * Nbatch;

    fid = fopen(ops.fproc, 'r'); % open the preprocessed data file

    k = 0;
    dd = gpuArray.zeros(ops.nt0, 5e4, 'single'); % preallocate matrix to hold 1D spike snippets
    if ops.fig % PLOTTING
        figure(1); hold on;
    end
    for ibatch = 1:nskip:Nbatch
        offset = 2 * ops.Nchan * batchstart(ibatch);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [ops.Nchan NT], '*int16');
        dat = dat';

        % move data to GPU and scale it back to unit variance
        dataRAW = gpuArray(dat);
        dataRAW = single(dataRAW);
        dataRAW = dataRAW / ops.scaleproc;

        % find isolated spikes from each batch
        [row, col] = isolated_peaks_multithreshold(-abs(dataRAW), ops, ibatch);

        % for each peak, get the voltage snippet from that channel
        clips = get_SpikeSample(dataRAW, row, col, ops, 0);
        c = sq(clips(:, :));
        if ops.fig == 1 % PLOTTING
            plot(c)
        end
        if k + size(c, 2) > size(dd, 2)
            dd(:, 2 * size(dd, 2)) = 0;
        end

        dd(:, k + [1:size(c, 2)]) = c;
        k = k + size(c, 2);
        if k > 1e5
            break;
        end
    end
    fclose(fid);
    if ops.fig == 1 % PLOTTING
        title('local isolated spikes (1D voltage waveforms)');
    end
    % discard empty samples
    dd = dd(:, 1:k);
    % window definition
    % window = tukeywin(size(wTEMP, 1), 0.9);
    sigma_time = 0.125; % ms of the gaussian kernel to focus penalty on central region of waveforms
    sigma = ops.fs * sigma_time / 1000; % samples
    gaussian_window = gausswin(size(dd, 1), (size(dd, 1) - 1) / (2 * sigma));
    zeros_for_tukey = zeros(size(dd, 1), 1);
    percent_tukey_coverage = 80;
    partial_tukey_window = tukeywin(ceil(size(dd, 1) * percent_tukey_coverage / 100), 0.5);
    % put middle of partial tukey at the center of zeros_for_tukey
    zeros_for_tukey(ceil(size(dd, 1) / 2) - ceil(size(partial_tukey_window, 1) / 2) + ...
        1:ceil(size(dd, 1) / 2) + floor(size(partial_tukey_window, 1) / 2)) = partial_tukey_window;
    tukey_window = zeros_for_tukey;

    dd_windowed = dd .* tukey_window;
    % align max absolute peaks to the center of the template (ops.nt0min)
    [~, peak_indexes] = max(abs(dd_windowed), [], 1);
    spikes_shifts = peak_indexes - ops.nt0min;
    dd_aligned = gpuArray(nan(size(dd)));
    dd_windowed_aligned = gpuArray(nan(size(dd)));
    for i = 1:size(dd_windowed, 2)
        dd_windowed_aligned(:, i) = circshift(dd_windowed(:, i), -spikes_shifts(i));
        dd_aligned(:, i) = circshift(dd(:, i), -spikes_shifts(i));
    end
    dd_cpu = double(gather(dd_windowed_aligned));

    % PCA is computed on the windowed data
    [U, ~, ~] = svdecon(dd_cpu); % the PCs are just the left singular vectors of the waveforms
    % if ops.fig == 1 % PLOTTING
    %     figure(4); hold on;
    %     for i = 1:nPCs
    %         plot(U(:,i)+i*1);
    %     end
    %     title(strcat("Top ", num2str(nPCs), " PCs"));
    % end
    wPCA = gpuArray(single(U(:, 1:nPCs))); % take as many as needed
    % adjust the arbitrary sign of the first PC so its peak is downward
    wPCA(:, 1) =- wPCA(:, 1) * sign(wPCA(ops.nt0min, 1));

    use_kmeans = true;
    % initialize the template clustering
    if use_kmeans
        dd_pca = wPCA' * dd_aligned;
        % compute k-means clustering of the waveforms
        rng('default'); rng(1); % initializing random number generator for reproducibility
        % stream = RandStream('mlfg6331_64');  % Random number stream
        p = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(p)
            options = statset('UseParallel', 0);
            num_jobs = 12;
        else
            options = statset('UseParallel', 1); %'UseSubstreams', 1,'Streams', stream);
            num_jobs = p.NumWorkers;
        end
        % try to use all available jobs, but if it fails, halve the number of jobs and
        % if it still fails, halve it again, until it doesn't fail
        % if all else fails, just run it sequentially
        try
            [cluster_id, ~, ~, Dist_from_K] = kmeans(dd_pca', nPCs, 'Distance', 'sqeuclidean', ...
                'MaxIter', 10000, 'Replicates', num_jobs, 'Display', 'final', 'Options', options);
        catch
            disp('k-means failed in parallel, running sequentially instead')
            [cluster_id, ~, ~, Dist_from_K] = kmeans(dd_pca', nPCs, 'Distance', 'sqeuclidean', ...
                'MaxIter', 10000, 'Replicates', num_jobs, 'Display', 'final');
        end
        spikes = gpuArray(nan(size(dd_aligned)));
        number_of_spikes_to_use = nan(nPCs, 1);
        for K = 1:nPCs
            % use 90% of the closest spikes to the cluster center, to avoid outliers
            number_of_close_spikes = floor(sum(cluster_id == K) * 0.9);
            [~, min_dist_idxs] = mink(Dist_from_K(:, K), number_of_close_spikes);
            spikes_to_use = dd_aligned(:, cluster_id == K & ismember(1:length(cluster_id), min_dist_idxs)');
            number_of_spikes_to_use(K) = size(spikes_to_use, 2);
            % choose closest spikes to the cluster center
            spikes(:, sum(number_of_spikes_to_use(1:K - 1)) + ...
                1:sum(number_of_spikes_to_use, 'omitnan')) = spikes_to_use;
        end
        % drop nan values
        spikes = spikes(:, ~isnan(spikes(1, :)));

        % dbstop in extractTemplatesfromSnippets.m at 140
        total_num_spikes_used = sum(number_of_spikes_to_use);
        % take max of wave peaks
        [peak_amplitudes, ~] = max(spikes, [], 1);

        wave_choice_left_bounds = [1; cumsum(number_of_spikes_to_use)];
        wave_choice_right_bounds = wave_choice_left_bounds(2:end);
        wave_choice_left_bounds = wave_choice_left_bounds(1:end - 1);
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
        [max_peaks, max_larger_sorted_idx] = sort(max(dd(:, max_larger_mask)), 'descend');
        [min_peaks, min_larger_sorted_idx] = sort(min(dd(:, ~max_larger_mask)), 'descend');
        if ~isempty(max_peaks) && ~isempty(min_peaks)
            peak_amplitudes = [max_peaks, min_peaks];
        elseif ~isempty(max_peaks)
            peak_amplitudes = max_peaks;
        elseif ~isempty(min_peaks)
            peak_amplitudes = min_peaks;
        else
            error('No spikes found in the data!')
        end
        % sort the spikes by amplitude, with positive and negative peaks separate
        max_mask_idx = find(max_larger_mask);
        min_mask_idx = find(~max_larger_mask);
        idx = [max_mask_idx(max_larger_sorted_idx), min_mask_idx(min_larger_sorted_idx)];
        total_num_spikes_used = length(idx);

        % assign non-uniform wave choice boundaries based on the amplutide of the peaks
        fraction_of_N_peaks = ceil(0.02 * length(peak_amplitudes));
        % get even distribution of spike amplitudes, treating positive and negative peaks separately
        % also skip the first and last chunks to avoid outliers
        num_max_peak_boundaries = ceil(length(max_peaks) / length(peak_amplitudes) * nPCs);
        num_min_peak_boundaries = nPCs - num_max_peak_boundaries;
        if ~isempty(max_peaks) && ~isempty(min_peaks) % if there are both positive and negative peaks
            max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks, length(max_peaks))), ...
                max_peaks(end), num_max_peak_boundaries);
            min_peak_boundaries = linspace(min_peaks(1), min_peaks(end - min(fraction_of_N_peaks, ...
                length(min_peaks))), num_min_peak_boundaries);
            % combine the boundaries
            peak_boundaries = [max_peaks(1), max_peak_boundaries, min_peak_boundaries(2:end), ...
                                   min_peaks(end)];
        elseif ~isempty(max_peaks) % if there are only positive peaks
            max_peak_boundaries = linspace(max_peaks(min(fraction_of_N_peaks, length(max_peaks))), ...
                max_peaks(end), nPCs);
            peak_boundaries = [max_peaks(1), max_peak_boundaries, max_peaks(end)];
        elseif ~isempty(min_peaks) % if there are only negative peaks
            min_peak_boundaries = linspace(min_peaks(1), min_peaks(end - min(fraction_of_N_peaks, ...
                length(min_peaks))), nPCs);
            peak_boundaries = [min_peaks(1), min_peak_boundaries, min_peaks(end)];
        else
            error('No spikes found in the data!')
        end
        % find the closest peak to each boundary
        [~, wave_choice_boundaries] = min(abs(peak_amplitudes - peak_boundaries'), [], 2);
        % N_waves_between_choices = diff(wave_choice_boundaries);

        % assign uniform wave choice boundaries
        % uniform_wave_choice_boundaries = round(linspace(1, length(peaks), nPCs+1));

        % wave_choice_left_bounds = wave_choice_boundaries(1:end-1);
        group_percent_expansion = 8; % percent
        group_size_expansion = ceil(group_percent_expansion / 2/100 * length(peak_amplitudes));
        % define the boundaries, preventing negative values
        wave_choice_left_expand = max(wave_choice_boundaries - group_size_expansion, 1);
        wave_choice_left_bounds = wave_choice_left_expand(1:end - 1);
        wave_choice_right_expand = min(wave_choice_boundaries + group_size_expansion, ...
            length(peak_amplitudes));
        wave_choice_right_bounds = wave_choice_right_expand(2:end);
        N_waves_between_choices = wave_choice_right_bounds - wave_choice_left_bounds;
        disp(['Chunks overlap by ', num2str(group_percent_expansion), '%, which is ', ...
                  num2str(group_size_expansion), ' spikes'])
        disp('Number of spikes in each chunk: ')
        disp(N_waves_between_choices')
    end
    if ops.fig % PLOTTING
        figure(22);
        plot((1:length(peak_amplitudes)) * ops.nt0, peak_amplitudes, 'm'); hold on;
        for iPeak = 1:length(peak_amplitudes)
            if mod(iPeak, 2) == 0
                color = 'k';
                if use_kmeans
                    plot((-ops.nt0min + 1:ops.nt0min - 1) + iPeak * ops.nt0, spikes(:, iPeak)', ...
                        'DisplayName', num2str(iPeak), 'Color', color)
                else
                    plot((-ops.nt0min + 1:ops.nt0min - 1) + iPeak * ops.nt0, dd(:, idx(iPeak))', ...
                        'DisplayName', num2str(iPeak), 'Color', color)
                end
            end
        end
        title('peak amplitudes for each spike')
        % plot vertical lines at the boundaries
        for iBoundary = 1:length(wave_choice_left_bounds)
            % different color for each chunk, from magenta to cyan
            color = [1 - iBoundary / length(wave_choice_left_bounds), iBoundary / ...
                         length(wave_choice_left_bounds), 1];
            plot([1, 1] * wave_choice_left_bounds(iBoundary) * ops.nt0, ylim, ...
                'Color', color, 'LineWidth', 2)
            plot([1, 1] * wave_choice_right_bounds(iBoundary) * ops.nt0, ylim, ...
                'Color', color, 'LineWidth', 2)
        end
    end
    if use_kmeans
        wTEMP = spikes(:, wave_choice_left_bounds); % start with first spike in each cluster
    else
        plot((1:length(peak_amplitudes)) * ops.nt0, peak_amplitudes, 'm', 'LineWidth', 3);
        wTEMP = dd(:, idx(wave_choice_left_bounds)); % initialize with a smooth range of amplitudes
    end
    largest_CC_idx = 1;
    N_tries_for_largest_CC_idx_so_far = 0;
    best_CC_idxs = wave_choice_left_bounds;
    lowest_total_cost_for_each_chunk = 1e12 * ones(1, length(wave_choice_left_bounds));
    iter = 1;

    while iter < 2
        % replace with next wave in chunk to check correlation,
        % each wave index withing the chunk starting at the wave_choice_left_bounds
        if use_kmeans
            descriptor = 'k-means cluster';
            wTEMP(:, largest_CC_idx) = spikes(:, wave_choice_left_bounds(largest_CC_idx) + ...
                N_tries_for_largest_CC_idx_so_far);
        else
            descriptor = 'chunk';
            wTEMP(:, largest_CC_idx) = dd(:, idx(wave_choice_left_bounds(largest_CC_idx) + ...
                N_tries_for_largest_CC_idx_so_far));
        end
        % multiply waveforms by a Gaussian with the sigma value
        % this is to make the correlation more sensitive to the central shape of the waveform
        % wTEMP_for_corr = wTEMP .* gausswin(size(wTEMP, 1), (size(wTEMP, 1) - 1) / (2 * sigma));
        % align largest peak of each template to ops.nt0min before checking correlation to avoid
        % the correlation being sensitive to the alignment of the waveforms
        wTEMP_for_corr = wTEMP .* gaussian_window; % use window to focus on central region of waveforms
        CC = corr(wTEMP_for_corr);

        %% section to compute terms of the cost function
        % get residual of the waveform for this row of the CC matrix
        % sum the absolute value of the residual, scale by the absolute value of wTEMP_for_corr
        % this is to avoid using the waves with non-central shapes, by using it as a cost function
        wTEMP_gaussian_residual = sum(abs(wTEMP(:, largest_CC_idx) - ...
            wTEMP_for_corr(:, largest_CC_idx))) / sum(abs(wTEMP_for_corr(:, largest_CC_idx)));

        % compute the penalty for the highest single correlation in the CC matrix for this template choice
        if largest_CC_idx == 1
            highest_template_similarity_penalty = max(max(CC(largest_CC_idx, 2:end), ...
                max(CC(2:end, largest_CC_idx))));
        else
            highest_template_similarity_penalty = max(max(CC(1:largest_CC_idx - 1, largest_CC_idx)), ...
                max(CC(largest_CC_idx, 1:largest_CC_idx - 1)));
        end
        highest_template_similarity_penalty = max(highest_template_similarity_penalty, 0); % make sure it is not negative

        % compute sum of all similarities with other previous template choices
        % negative correlations are not penalized
        if largest_CC_idx == 1
            corr_sum_with_other_template_choices = sum(max(CC(largest_CC_idx, 2:end), 0)) + ...
                sum(max(CC(2:end, largest_CC_idx), 0));
        else
            corr_sum_with_other_template_choices = sum(sum(max(CC(1:largest_CC_idx - 1, largest_CC_idx), 0))) + ...
                sum(sum(max(CC(largest_CC_idx, 1:largest_CC_idx - 1), 0)));
        end

        % compute the total cost for this wave choice, cheapest waveform will be chosen
        corr_sum_with_other_template_choices_term = corr_sum_with_other_template_choices;
        wTEMP_gaussian_residual_term = nPCs * wTEMP_gaussian_residual;
        highest_template_similarity_penalty_term = nPCs * highest_template_similarity_penalty;
        total_cost_for_wave = corr_sum_with_other_template_choices_term + ...
            wTEMP_gaussian_residual_term + highest_template_similarity_penalty_term;

        % choose the best wave for this chunk/cluster
        % check the cost, and ensure that the wave has not already been chosen before in a previous chunk
        if (total_cost_for_wave < lowest_total_cost_for_each_chunk(largest_CC_idx)) && ...
                ~ismember(wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far, best_CC_idxs)
            lowest_total_cost_for_each_chunk(largest_CC_idx) = total_cost_for_wave;
            best_CC_idxs(largest_CC_idx) = wave_choice_left_bounds(largest_CC_idx) + N_tries_for_largest_CC_idx_so_far;
        end
        N_tries_for_largest_CC_idx_so_far = N_tries_for_largest_CC_idx_so_far + 1;
        % terminate if we have tried all waves in the amplitude-sorted chunk
        if N_tries_for_largest_CC_idx_so_far >= N_waves_between_choices(largest_CC_idx) || ...
                N_tries_for_largest_CC_idx_so_far >= total_num_spikes_used
            disp(strcat("Tried all waves in "+descriptor + " ", num2str(largest_CC_idx), ...
                ", using wave idx with best CC: ", num2str(best_CC_idxs(largest_CC_idx))))
            % sorted_CC = sort(CC(largest_CC_idx,:), 'descend');
            disp(strcat("Total cross-channel correlation for this cluster ", num2str(sum(abs(CC(largest_CC_idx, :))))))
            disp("Residual cost for this "+descriptor + " was " + num2str(wTEMP_gaussian_residual))
            disp("Highest template similarity penalty for this "+descriptor + " was " + num2str(highest_template_similarity_penalty))
            disp("Percent of influence for each term: ")
            corr_sum_with_other_template_choices_percent = ...
                corr_sum_with_other_template_choices_term / total_cost_for_wave * 100;
            wTEMP_gaussian_residual_percent = wTEMP_gaussian_residual_term / total_cost_for_wave * 100;
            highest_template_similarity_penalty_percent = ...
                highest_template_similarity_penalty_term / total_cost_for_wave * 100;
            disp([corr_sum_with_other_template_choices_percent, wTEMP_gaussian_residual_percent, ...
                      highest_template_similarity_penalty_percent])
            disp("Final cost for this "+descriptor + " was: " + num2str(lowest_total_cost_for_each_chunk(largest_CC_idx)))
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

    if use_kmeans
        wTEMP = dd(:, randperm(size(dd, 2), nPCs)); % removing this line will cause KS to not find
        % spikes sometimes... variable is overwritten in the next line, so it's inconsequential
        wTEMP(:, 1:nPCs) = spikes(:, best_CC_idxs(1:nPCs));
    else
        wTEMP(:, 1:nPCs) = dd(:, idx(best_CC_idxs(1:nPCs)));
    end

    wTEMP = wTEMP ./ sum(wTEMP .^ 2, 1) .^ .5; % normalize the templates
    if ops.fig % PLOTTING
        % wTEMP_for_CC_final = wTEMP .* repmat(gausswin(size(wTEMP,1), (size(wTEMP,1)-1)/(2*sigma)), 1, size(wTEMP,2));
        % specify colormap to be 'cool'
        cmap = colormap(cool(nPCs));
        figure(2); hold on;
        scale = 0.8;
        for i = 1:nPCs
            windowed_wTEMP = tukey_window .* wTEMP(:, i);
            plot(wTEMP(:, i) + i * scale, 'LineWidth', 2, 'Color', cmap(i, :));
            % plot standardized Gaussian multiplied waveforms for comparison
            plot(windowed_wTEMP + i * scale, 'r');
            if i == nPCs
                % show gaussian window
                plot(i * scale + 0.5 * tukey_window, 'g');
                plot(i * scale + 0.5 * gaussian_window, 'c');
            end
        end
        title('initial templates');
        pbaspect([1 2 1])
    end

    % use k-means isolated spikes for correlation calculation and averaging to ignore outlier spikes
    if use_kmeans
        % just take average of top 10 percent most correlated spikes to the ones chosen in wTEMP
        % for each cluster. Use wave_choice_left_bounds to get the spikes to use from 'spikes'
        for iCluster = 1:nPCs
            % get the spikes that were chosen for this cluster
            cluster_spikes = spikes(:, wave_choice_left_bounds(iCluster):wave_choice_right_bounds(iCluster));
            % get the correlation matrix for this cluster
            cluster_CC = corr(cluster_spikes);
            % get the top 1 percent most correlated spikes to the chosen spikes, must be at least 10 spikes
            try
                [~, top_CC_idxs] = maxk(cluster_CC(:, 1), max(ceil(size(cluster_CC, 1) * 0.01), 10));
            catch
                % if there are less than 10 spikes in the cluster, just take the top 1 percent
                [~, top_CC_idxs] = maxk(cluster_CC(:, 1), ceil(size(cluster_CC, 1) * 0.01));
                
            end

            % get the top 10 percent most correlated spikes
            top_CC_spikes = cluster_spikes(:, top_CC_idxs);
            % average the top 10 most correlated spikes
            wTEMP(:, iCluster) = mean(top_CC_spikes, 2);
        end

        wTEMP = wTEMP ./ sum(wTEMP .^ 2, 1) .^ .5; % standardize the new clusters
    else % use all spikes for correlation calculation and averaging
        for i = 1:10
            % at each iteration, assign the waveform to its most correlated cluster
            CC = wTEMP' * dd;
            [amax, imax] = max(CC, [], 1); % find the best cluster for each waveform
            for j = 1:nPCs
                wTEMP(:, j) = dd(:, imax == j) * amax(imax == j)'; % weighted average to get new cluster means
            end
            wTEMP = wTEMP ./ sum(wTEMP .^ 2, 1) .^ .5; % standardize the new clusters
        end
    end
    % tukey it
    wTEMP_tukeyed = wTEMP .* tukey_window;
    if ops.fig % PLOTTING
        figure(3); hold on;
        for i = 1:nPCs
            plot(wTEMP(:, i) + i * scale, 'LineWidth', 2, 'Color', cmap(i, :));
            plot(wTEMP_tukeyed(:, i) + i * scale, 'r');
            if i == nPCs
                % show tukey
                plot(i * scale + 0.5 * tukey_window, 'c');
            end
        end
        % set aspect ratio to 3, 1
        title('prototype templates');
        pbaspect([1 2 1])
    end

    if use_kmeans
        % recomputing PCA on the k-means isolated spikes (90% of closest spikes to cluster center)
        [U, ~, ~] = svdecon(spikes);
        wPCA = gpuArray(single(U(:, 1:nPCs))); % take as many as needed
    end
end
