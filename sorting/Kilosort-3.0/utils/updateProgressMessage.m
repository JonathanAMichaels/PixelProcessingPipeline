function updateProgressMessage(n, ntot, tbase, len, freq)

%% Defaults
if nargin<5 || isempty(freq)
    freq = 1; % freq of full message updates; else just print '.' on each call
end

if nargin<4 || isempty(len)
    len = 100; % padded string length
end

if nargin<3 || isempty(tbase)
    t = toc;
else
    t = toc(tbase);
end


%% Make string

if mod(n, freq)>0 && n~=ntot
    % do nothing...just passing through
    fprintf('.')
    return
else
    % Create progress message & print to command window
    % update times
    if t<90
        tstr = sprintf('%2.2f sec elapsed', t);
    else
        tstr = sprintf('%2.2f min elapsed', t/60);
    end
    secPerN = t/n;
    tRemEstimate = secPerN * (ntot-n);
    if tRemEstimate<90
        rstr = sprintf('%2.2f sec remaining', tRemEstimate);
    else
        rstr = sprintf('%2.2f min remaining', tRemEstimate/60);
    end
    % update message
    msg = sprintf('\nfinished %4i of %i.  (%s, ~%s; %2.2f sec/each)..',n, ntot, tstr, rstr, secPerN);

    % clear previous message
    if n>freq
        fprintf(repmat('\b',1,len + freq-1));
    end
    % print message
    fprintf(pad(msg, len, '.'));

end
    
end %main function

