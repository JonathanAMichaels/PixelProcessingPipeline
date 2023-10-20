function [row, col, mu] = isolated_peaks_buffered_czuba(S1, ops, ibatch)
    % takes a matrix of timepoints by channels S1
    % outputs threshold crossings that are relatively isolated from other peaks
    % outputs row, column and magnitude of the threshold crossing
    %
    % 2021-04-13  TBC  Corrected buffer exclusion param (was [nt0] should be [ntbuff])
    %                  Renamed isolated_peaks_new.m to isolated_peaks_buffered.m
    %                  - incase instances of non-buffered input data are needed
    % 2023-10-14  SMO  Added multi-thresholding capability, to get a wide range of spikes for
    %                  different amplitude MUs. Previously, when the threshold was set to a
    %                  single value, the algorithm would only find spikes with that amplitude
    %                  because crowding would occur and large spikes would never match the
    %                  isolation criteria. Now all matches are appended to the outputs.

    loc_range = getOr(ops, 'loc_range', [5 4]);
    long_range = getOr(ops, 'long_range', [30 6]);
    Th = ops.spkTh;
    ntbuff = ops.ntbuff;
    full_row_cell = cell(length(Th), 1);
    full_col_cell = cell(length(Th), 1);
    full_mu_cell = cell(length(Th), 1);
    channel_offset = 20;
    time_offset = (ibatch - 1) * size(S1, 1);
    for iTh = 1:length(Th)
        % finding the local minimum in a sliding window within plus/minus loc_range extent
        % across time and across channels
        smin = my_min(S1, loc_range, [1 2]);
        peaks = single(S1 < smin +1e-3 & S1 < Th(iTh)); % the peaks are samples that achieve this local minimum, AND have negativities less than a preset threshold

        % only take local peaks that are isolated from other local peaks
        sum_peaks = my_sum(peaks, long_range, [1 2]); % if there is another local peak close by, this sum will be at least 2
        peaks = peaks .* (sum_peaks < 1.2) .* S1; % set to 0 peaks that are not isolated, and multiply with the voltage values

        % exclude temporal buffers
        % - [nt0] is NOT the temporal buffer param, should be [ntbuff]
        peaks([1:ntbuff end - ntbuff:end], :) = 0;

        [row, col, mu] = find(peaks); % find the non-zero peaks, and take their amplitudes
        mu =- mu; % invert the sign of the amplitudes

        full_row_cell{iTh} = row;
        full_col_cell{iTh} = col;
        full_mu_cell{iTh} = mu;

        % % plot but offset the time by batch width for each ibatch, make color of peak label match the threshold value
        % if ops.fig && (mod(ibatch, 20) == 0 || ibatch == ops.Nbatch-1)
        %     figure(999); hold on;
        %     for i = 1:size(S1, 2)
        %         time = (1:size(S1, 1)) + time_offset;
        %         plot(time, S1(:, i) + i * channel_offset, 'k');
        %     end
        %     % color cyan to magenta
        %     plot(row + time_offset, col * channel_offset - mu, '*', 'Color', [1 - iTh / length(Th), 0, iTh / length(Th)]);
        % end
    end
    full_row = vertcat(full_row_cell{:});
    full_col = vertcat(full_col_cell{:});
    full_mu = vertcat(full_mu_cell{:});
    if isempty(full_row) || isempty(full_col) || isempty(full_mu)
        row = double.empty(0, 1);
        col = double.empty(0, 1);
        mu = double.empty(0, 1);
        return;
    end
    % if any duplicate times exist, take from the channel with the largest amplitude, using argmax
    row_uniq = unique(full_row);
    times_which_have_duplicates = row_uniq(histcounts(full_row, [row_uniq; row_uniq(end) + 1]) > 1);
    dups_to_remove = [];
    for iDup = 1:length(times_which_have_duplicates)
        % find the indices of the duplicates
        dup_inds = find(full_row == times_which_have_duplicates(iDup));
        % find the index of the max amplitude
        [~, max_ind] = max(full_mu(dup_inds));
        % remove all but the max amplitude
        dup_inds(max_ind) = [];
        dups_to_remove = [dups_to_remove; dup_inds];
    end
    full_row(dups_to_remove) = [];
    full_col(dups_to_remove) = [];
    full_mu(dups_to_remove) = [];
    if length(row_uniq) ~= length(full_row)
        error('Duplicate removal failed, numbers of unique spikes and spikes left after removal do not match.');
    end
    % disp(['Removed ' num2str(length(dups_to_remove)) ' duplicate spikes, out of ' num2str(length(full_row)) ' total spikes.']);

    % now plot the remaining spikes at each amplitude with red squares
    % if ops.fig && (mod(ibatch, 20) == 0 || ibatch == ops.Nbatch)
    %     figure(999); hold on;
    %     plot(full_row + time_offset, full_col * channel_offset - full_mu, 'rs');
    % end
    row = full_row;
    col = full_col;
    mu = full_mu;
end
