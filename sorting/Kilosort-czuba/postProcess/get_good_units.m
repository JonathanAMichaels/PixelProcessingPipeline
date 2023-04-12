function [igood, status] = get_good_units(rez)

NN = size(rez.dWU,3);
sd = zeros(NN, 1); 
for k = 1:NN
    wav = rez.dWU(:, :, k);
    mwav = sq(sum(wav.^2, 1));
    
%     wav = sq(rez.U(:, k, :));
%     mwav = sq(sum(wav.^2, 2));
    
    mmax = max(mwav);
    mwav(mwav<mmax/10) = 0;
    
    xm = mean(rez.xc(:) .* mwav(:)) / mean(mwav);
    ym = mean(rez.yc(:) .* mwav(:)) / mean(mwav);
    
    ds = sqrt((rez.xc(:) - xm).^2 + (rez.yc(:) - ym).^2);
    sd(k) = gather(mean(ds(:) .* mwav(:))/mean(mwav));
    
end
if isfield(rez,'good')
    igood = rez.good & sd<100;
else
    igood = sd<100;
end    
igood = double(igood);

statStr = sprintf('Found %d good units (of %d total)', sum(igood>0), numel(igood));
if nargout<2
    cmdLog(statStr)
else
    status = statStr;
end

end %main function
