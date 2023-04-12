function addFigInfo(ops, H)

if nargin<2
    H = gcf;
end

try
    figure(H);
    fsz = 10; % info font size
    % date of Kilosort processing
    if isfield(ops, 'datenumSorted') && ~isempty(ops.datenumSorted)
        dateVer = sprintf('Sorted on: %s',datestr(ops.datenumSorted));
    else
        dateVer = sprintf('Sorted on: %s',datestr(now));
    end
    
    try
        % append kilosort git repo information
        gitstat = strsplit(ops.git.kilosort.status, '\n');
        dateVer = [dateVer, sprintf('        Kilosort git: %s,  commit %s', gitstat{1}, ops.git.kilosort.revision(1:7))];
    end
    infostr = {dateVer, ...
               sprintf('Raw data:\t%s',ops.fbinary), ...
               sprintf('Output dir:\t%s',ops.saveDir)};
    % add axis for text info
    axes('position',[0,.002,1,.02],'visible','off');
    % shrinking text if multiple lines
    if contains(infostr, {sprintf('\n'),sprintf('\n\r')}), fsz = 8; end
    text(0,0, infostr, 'verticalAlignment','bottom', 'interpreter','none', 'fontsize',fsz);
catch ME
    warning(ME.identifier,'Error labeling figure was: %s',ME.message);
end