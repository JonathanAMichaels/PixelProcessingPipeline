function spike_validation_plot(chunk, clusters)
    script_dir = pwd; % get directory where repo exists
    load(fullfile(script_dir, '/tmp/config.mat'))
    load(fullfile(myo_sorted_dir, 'brokenChan.mat'))
    disp(['Using this channel map: ' myo_chan_map_file])
    % load channel map with broken channels removed if chosen by user
    if length(brokenChan) > 0 && remove_bad_myo_chans(1) ~= false
        load(fullfile(myo_sorted_dir, 'chanMap_minus_brokenChans.mat'))
    else
        load(myo_chan_map_file)
    end

    close all
    dbstop if error

    chunk_index_range = get_data_amount(chunk);
    disp(['Plotting spike validation plot for Chunk ' num2str(chunk) ', with range of indices: ' num2str(chunk_index_range(1)) ' to ' num2str(chunk_index_range(end))])

    processed_ephys_data_path = [myo_sorted_dir '/proc.dat'];
    final_sort_path = [myo_sorted_dir '/custom_merges/final_merge/custom_merge.mat'];
    channels = chanMap;

    % Input: provide the path to the custom_merge.mat file.
    fid = fopen(processed_ephys_data_path,'r');
    data_1D = fread(fid, 'int16');
    fclose(fid);
    load(final_sort_path,'C','I','T','mdata');  
    spike_times = T;
    cluster_ID = I;
    mdata_full = mdata;
    if isa(clusters, 'logical') && clusters == true
        disp("Showing all clusters.")
    else
        C = intersect(C, clusters+1);
        disp("Showing clusters: " + num2str(clusters+1))
    end
    mdata = mdata_full(:,:,C);
    mdata_size = size(mdata_full);
    template_width = mdata_size(1);
    num_chans = length(channels);

    dbstop if errors
    data = reshape(data_1D,num_chans,length(data_1D)/num_chans)';
    data = data(:,channels);
    chan_cmap = gray(32);
    chan_cmap = repmat(chan_cmap(16:32)',3); % get lighter half
    clust_cmap = prism(length(C)); % get full rainbow

    %     figure(1)
    %     histogram(I, 'FaceColor', [0 0 0]);
    %     title('Spike Counts for Each Template ID')
    %     xlabel('Template ID')
    %     ylabel('Spike Counts')
    
    figure('CloseRequestFcn',@my_closereq); hold on
    data_mins = min(data(chunk_index_range,:));
    data_maxs = max(data(chunk_index_range,:));
    data_ranges = data_maxs - data_mins;
    norm_data = (data(chunk_index_range,:)).*(1./data_ranges);

    temp_mins = min(mdata(:,:,:));
    temp_maxs = max(mdata(:,:,:));
    temp_ranges = temp_maxs - temp_mins;
    norm_temp_rngs = repmat(max(temp_ranges),1,length(channels));
    norm_temp = (mdata(:,:,:)).*(1./norm_temp_rngs);
    data_amount_size = length(chunk_index_range);

    %     disp_ch = 
    for jj = 1:length(channels)
    %         ch = channels(jj);
        plot(chunk_index_range, norm_data(:,jj)+2*jj*ones(data_amount_size,1), ...
            'color',chan_cmap(jj,:), ...
            'LineWidth',1.2)
    end

    for ii = 1:length(C)
        cc = C(ii);
        bitmask = ismember(cluster_ID,cc);
        spikes_for_cluster = spike_times(bitmask);
        trunc_idxs = spikes_for_cluster<max(chunk_index_range) & spikes_for_cluster>min(chunk_index_range);
        trunc_spike_times = spikes_for_cluster(trunc_idxs);
    %         trunc_cluster_ID = cc.*uint32(ones(sum(trunc_idxs),1));
    %        s = scatter(trunc_spike_times,trunc_cluster_ID,'|');
        clust_template = norm_temp(:,:,ii);
        for kk = 1:length(channels)
    %             ch = channels(kk);
            for iT=1:length(trunc_spike_times)-1
                plot( ...
                    (trunc_spike_times(iT)-floor(template_width/2)+1):(trunc_spike_times(iT)+floor(template_width/2)), ...
                    clust_template(:,kk)+1+kk*2, ...
                    'color',[clust_cmap(ii,:) 0.5],...
                    'LineWidth',1.2);
    %                 t = plot(trunc_spike_times,trunc_cluster_ID,'|');
            end
        end
        alpha(0.2)
    %         set(s(1), ...
    %             'SizeData', 500, ...
    %             'LineWidth',1.5, ...
    %             'MarkerEdgeColor', clust_cmap(ii,:), ...
    %             'MarkerEdgeAlpha', 0.5)
    end
    title('Template Matches for Each Channel')
    xlabel('Time (s)')
    ylabel('Neural Activity (a.u.)')
    ax = gca;
    % ax.XAxis.Exponent = 0;
    set(gcf, 'WindowState', 'fullscreen'); % set fullscreen
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    set(ax,'color',[0 0 0])
    set(ax, 'YTick', []);
    set(ax, 'XTick', chunk_index_range(1):30000:chunk_index_range(end));
    set(ax, 'XTickLabel', chunk_index_range(1)/30000:1:chunk_index_range(end)/30000);
end

% get 10 second chunks of data
function data_amount = get_data_amount(chunk)
    data_amount = (chunk-1)*300000+1:chunk*300000;
end

function my_closereq(src,event)
    % Close request function 
    % to quit MATLAB when plot is closed
    disp('Plot closed. Quitting MATLAB.')
    delete(gcf)
    quit
end