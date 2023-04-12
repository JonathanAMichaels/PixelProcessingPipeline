function make_fig(W, U, mu, nsp, ibatch)

try
    % wrap in try/catch just incase figure gets closed *while executing* (...because happened)
    
if nargin<5 || isempty(ibatch)
    batchStr = '';
else
    batchStr = sprintf('batch(%d)  || ',ibatch);
end
% Moved spatial & temporal template images to leftmost panels for easier comparison

% xy limits
ampLim = [5 1.02*gather(max(mu(:)))]; % spike amplitude, lin scaled
nspLim = [0.95, ceil(1.1*gather(max(nsp(:))))]; % spike count, log scaled
nchan = size(U,1);
[~, peakCh] = max(gather(abs(U(:,:,1))));

subplot(2,2,1)
imagesc(W(:,:,1))
title([batchStr, 'Temporal Components'])
xlabel('Unit number');
ylabel('Time (samples)');
set(gca,'xgrid','on')

subplot(2,2,3)
imagesc(U(:,:,1))
title('Spatial Components')
xlabel('Unit number');
ylabel('Channel number');
set(gca, 'xgrid','on', 'ygrid','on')

subplot(2,2,2)
plot(mu)
ylim(ampLim)
title('Unit Amplitudes')
xlabel('Unit number');
ylabel('Amplitude (arb. units)');

Hsp = subplot(2,2,4);
% semilogx(1+nsp, mu/2, '.')
% ylim(ampLim)
% xlim(nspLim)
% title('Amplitude vs. Spike Count')
% xlabel('Spike Count');
% ylabel('Amplitude (arb. units)');
if 0
    s = scatter(peakCh, mu, log2(gather(nsp+1)).*30+5, nsp, 'o','filled', 'MarkerFaceAlpha',0.8);
    set(gca,'yscale','linear')%, 'clim', [0,max(gather(mu))])
    ylim(ampLim);
    xlim([0, size(U,1)+1]);
    title({'Spike Count & Amplitude (size/color)','vs. Peak Channel'})
    xlabel('Peak channel');
    ylabel('Amplitude (arb. units)');
    cm = flipud(gray(32)); cm = cm(6:end,:);
    colormap(gca, cm);
    cb = colorbar;
    set(cb.Label, 'string','spike count')
%     s = scatter(peakCh, 1+nsp, (mu/2).^2, mu, 'o','filled', 'MarkerFaceAlpha',0.4);
%     set(gca,'yscale','log', 'clim', [0,max(gather(mu))])
%     ylim(nspLim);
%     xlim([0, size(U,1)+1]);
%     title({'Spike Count & Amplitude (size)','vs. Peak Channel'})
%     xlabel('Peak channel');
%     ylabel('Spike Count');
%     colormap(gca, flipud(gray(16)));
%     colorbar;
else
    scatter(1+gather(nsp), gather(mu), 70, peakCh', 'o','filled', 'MarkerFaceAlpha',0.8);
    cl = [1,nchan*1.2];
    set(Hsp,'xscale','log', 'clim', cl);
    ylim(ampLim);
    xlim(nspLim);
    title({'Amplitude vs. Spike Count','colored by Peak Channel'})
    xlabel('Spike Count');
    ylabel('Amplitude (arb. units)');
    colormap(Hsp, parula(nchan));
    cb = colorbar;
end
set(cb.Label, 'string','ch #')

drawnow

end %end try/catch

end %main function