function [rez, spike_times_for_kid] = template_learning(rez, tF, st3)

    wPCA = rez.wPCA; % shape is #PC components, #channels
    iC = rez.iC;
    ops = rez.ops;

    xcup = rez.xcup;
    ycup = rez.ycup;

    wroll = [];
    tlag = [-2, -1, 1, 2];
    % compute a product of each PC component with itself at 4 different time lags
    for j = 1:length(tlag)
        wroll(:, :, j) = circshift(wPCA, tlag(j), 1)' * wPCA;
    end

    %% split templates into batches by grid location
    rmin = 0.6; % minimum correlation between templates
    nlow = 100; % minimum number of spikes needed to keep a template
    n0 = 0; % number of clusters so far
    use_CCG = 0;

    Nchan = rez.ops.Nchan;
    Nk = size(iC, 2);
    yunq = unique(rez.yc);

    ktid = int32(st3(:, 2)) + 1; % get upsampled grid location of each spike
    tmp_chan = iC(1, :);
    ss = double(st3(:, 1)) / ops.fs; % spike times in seconds

    dmin = rez.ops.dmin;
    ycenter = (min(rez.yc) + dmin - 1):(2 * dmin):(max(rez.yc) + dmin + 1);
    dminx = rez.ops.dminx;
    xcenter = (min(rez.xc) + dminx - 1):(2 * dminx):(max(rez.xc) + dminx + 1);
    [xcenter, ycenter] = meshgrid(xcenter, ycenter); % define grid of electrode locations
    xcenter = xcenter(:);
    ycenter = ycenter(:);

    Wpca = zeros(ops.nPCs, Nchan, 1000, 'single');
    spike_times_for_kid = cell(1000, 1);
    nst = numel(ktid); % number of spikes
    hid = zeros(nst, 1, 'int32');

    % flag for plotting
    ops.fig = getOr(ops, 'fig', 1);

    tic
    for j = 1:numel(ycenter) % process spikes found for each y grid location
        if rem(j, round(numel(ycenter) / 10)) == 0 % print progress at most 10 times
            fprintf('time %2.2f, grid loc. grp. %d/%d, units %d \n', toc, j, numel(ycenter), n0)
        end

        y0 = ycenter(j); % get y electrode locations
        x0 = xcenter(j); % get x electrode locations
        % exclude grid locations that are not by electrodes
        xchan = (abs(ycup - y0) < dmin) & (abs(xcup - x0) < dminx);
        itemp = find(xchan); % get nearby electrode location indices for templates

        if isempty(itemp)
            continue;
        end
        tin = ismember(ktid, itemp); % get bitmask for spikes near electrodes
        pid = ktid(tin); % get spike ids for spikes near electrodes
        data = tF(tin, :, :); % exclude PC convolutions for spikes far from electrodes

        if isempty(data)
            continue;
        end
        %     size(data)

        %https://github.com/MouseLand/Kilosort/issues/427
        try
            ich = unique(iC(:, itemp));
        catch
            tmpS = iC(:, itemp);
            ich = unique(tmpS);
        end
        %     ch_min = ich(1)-1;
        %     ch_max = ich(end);

        if numel(ich) < 1
            continue;
        end

        nsp = size(data, 1);
        dd = zeros(nsp, ops.nPCs, numel(ich), 'single'); % #spikes, #PC components, #channels
        for k = 1:length(itemp) % for each template
            ix = pid == itemp(k); % ix is bitmask for spikes near this electrode
            % how to go from channels to different order, ib is indeces ordered like ich
            [~, ia, ib] = intersect(iC(:, itemp(k)), ich);
            dd(ix, :, ib) = data(ix, :, ia); % dd is just the PC convolutions ordered by distance from electrode
        end

        kid = run_pursuit(dd, nlow, rmin, n0, wroll, ss(tin), use_CCG, ops.nPCs);

        [~, ~, kid] = unique(kid); % make cluster ids consecutive
        nmax = max(kid); % number of clusters found
        for t = 1:nmax % for each cluster
            %         Wpca(:, ch_min+1:ch_max, t + n0) = gather(sq(mean(dd(kid==t,:,:),1)));
            % compute mean PC coordinates for each cluster of spikes, there is a separate PC space for each channel
            Wpca(:, ich, t + n0) = gather(sq(mean(dd(kid == t, :, :), 1)));
            spike_times_for_kid{t + n0} = st3(tin(kid == t), 1); % get spike times for each cluster
        end

        hid(tin) = gather(kid + n0);
        n0 = n0 + nmax;
    end
    Wpca = Wpca(:, :, 1:n0);
    % Wpca = cat(2, Wpca, zeros(size(Wpca,1), ops.nEig-size(Wpca, 2), size(Wpca, 3), 'single'));
    spike_times_for_kid = spike_times_for_kid(1:n0);
    % plot mean PC coordinates for each cluster for each channel and cluster (not that useful)
    % if ops.fig
    %     ichc = gather(ich);
    %     nPCs = size(Wpca, 1);
    %     figure(12)
    %     for k=1:n0; for ichcs=1:numel(ichc); for iPC=1:nPCs; scatter(ichcs*ones('like',Wpca(:,ichcs,k)), Wpca(:,ichcs,k)+40*k); end; end; end
    %     % plot top 3 PC coordinates for each cluster for all channels in each cluster
    %     figure(13)
    %     for k=1:n0
    %         scatter3(Wpca(1,1,k), Wpca(2,1,k), Wpca(3,1,k), 20, [1-k/n0,k/n0,k/n0]);
    %         hold on;
    %     end
    %     xlabel('PC1');
    %     ylabel('PC2');
    %     zlabel('PC3');
    %     set(gca,'DataAspectRatio',[1 1 1]);
    % end
    toc
    %%
    % Ncomps = min(ops.nEig, size(Wpca, 2));
    Ncomps = ops.nEig;
    rez.W = zeros(ops.nt0, 0, Ncomps, 'single');
    rez.U = zeros(ops.Nchan, 0, Ncomps, 'single');
    rez.mu = zeros(1, 0, 'single');
    figure(14); hold on;
    RGB_colors = rand(n0, 3);
    for t = 1:n0 % for each cluster
        dWU = wPCA * gpuArray(Wpca(:, :, t)); % multiply PC components by mean PC coordinates for each cluster
        [w, s, u] = svdecon(dWU); % compute SVD of that product to deconstruct it into spatial and temporal components
        wsign = -sign(w(ops.nt0min, 1)); % flip sign of waveform if necessary, for consistency
        % vvv save first Ncomps components of W, containing final rotation matrix
        rez.W(:, t, :) = gather(wsign * w(:, 1:Ncomps));
        % vvv save first Ncomps components of U, containing initial rotation and scaling matrix
        rez.U(:, t, :) = gather(wsign * u(:, 1:Ncomps) * s(1:Ncomps, 1:Ncomps));
        rez.mu(t) = gather(sum(sum(rez.U(:, t, :) .^ 2)) ^ .5); % get norm of U
        rez.U(:, t, :) = rez.U(:, t, :) / rez.mu(t); % normalize U
        if ops.fig
            for iloc = 1:numel(ich) % for each channel
                % plot PC component reconstructions for each channel, with color based on cluster
                plot(dWU(:, iloc) - 10 * iloc, 'color', RGB_colors(t, :));
            end
        end
    end

    if ops.fig
        title('First Multi-Channel Templates (Color Coded by Cluster)');
        xlabel('Time');
        ylabel('Channel');
    end
    %%
    rez.ops.wPCA = wPCA;
    % remove any NaNs from rez.W
    rez.W(isnan(rez.W)) = 0;
end
