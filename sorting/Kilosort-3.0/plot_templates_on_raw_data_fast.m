
function plot_templates_on_raw_data_fast(rez, st3)
    ops = rez.ops;
    fid = fopen(ops.fproc, 'r'); % open the preprocessed data file
    Nbatch = rez.temp.Nbatch;
    NT = ops.NT;
    batchstart = 0:NT:NT * Nbatch;
    Nfilt = size(rez.W, 2);
    nt0 = ops.nt0;
    Nchan = ops.Nchan;
    RGBA_colors = [rand(Nfilt, 3) 0.7*ones(Nfilt, 1)];
    for ibatch = 2:2:4
        offset = 2 * ops.Nchan * batchstart(ibatch);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [ops.Nchan NT], '*int16');
        dat = dat';

        % move data to GPU and scale it back to unit variance
        dataRAW = dat;
        dataRAW = single(dataRAW);
        dataRAW = dataRAW / ops.scaleproc; % dataRAW is size 
        
        % add offset to each channel, shift time to correct batch offset, then plot
        spacing = 30;
        dataRAW = dataRAW + spacing*ops.chanMap';
        batch_time = (1:size(dataRAW, 1)) + batchstart(ibatch);
        figure(15+round(ibatch/2)); hold on;
        % plot the raw data
        plot(repmat(batch_time', 1, size(dataRAW, 2)), dataRAW, 'k', 'LineWidth', 1);
        
        
        % next plot each template for each cluster, WU(:,:,j), on top of the raw data at each cluster's corresponding spike time
        % use only valid channel locations and scale the color by cluster ID, along RGB
        spike_times_in_batch_for_each_cluster = cell(Nfilt, 1);
        for jfilt = 5:10 % only show first four clusters %Nfilt
            % WU(:, :, j) = rez.mu(j) * squeeze(rez.W(:, j, :)) * squeeze(rez.U(:, j, :))';
            spike_times_in_batch_for_this_cluster = st3(st3(:, 2) == jfilt & st3(:, 1) > batchstart(ibatch) & st3(:, 1) < batchstart(ibatch+1), 1);
            Nspikes = length(spike_times_in_batch_for_this_cluster);
            if Nspikes == 0
                continue
            end
            spike_times_in_batch_for_each_cluster{jfilt} = spike_times_in_batch_for_this_cluster;
            unit_var_cluster_waveforms = rez.dWU(:, :, jfilt)./std(rez.dWU(:, :, jfilt));
            % get the 1D waveform for each channel in WU, and plot it at the corresponding spike time and channel location
            cluster_waveforms_for_all_channels = unit_var_cluster_waveforms + spacing*ops.chanMap';
            % create a time range centered on the spike time for all spikes, put result in 2D matrix
            time_ranges_for_each_spike_time = repmat((-nt0/2:nt0/2-1)', 1, Nspikes);
            offset_time_ranges_for_each_spike_time = time_ranges_for_each_spike_time + spike_times_in_batch_for_this_cluster';
            offset_time_ranges_for_each_spike_time_rep = repmat(offset_time_ranges_for_each_spike_time, 1, Nchan);
            cluster_waveforms_for_all_channels_rep = zeros(size(offset_time_ranges_for_each_spike_time_rep));
            for ichan = 1:Nchan
                cluster_waveforms_for_all_channels_rep(:, ((ichan-1)*Nspikes:ichan*Nspikes-1)+1) = repmat(cluster_waveforms_for_all_channels(:, ichan), 1, Nspikes);
            end
            % plot all waveforms for this cluster
            plot(offset_time_ranges_for_each_spike_time_rep, cluster_waveforms_for_all_channels_rep, 'Color',RGBA_colors(jfilt,:));
        end % rez.dWU has shape nt0 by Nchan by Nfilt
        title(["Prototype template matches for batch ", num2str(ibatch)])
    end
end