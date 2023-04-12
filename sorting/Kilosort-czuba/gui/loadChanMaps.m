function chanMaps = loadChanMaps()

ksroot = fileparts(fileparts(mfilename('fullpath')));
chanMapFiles = dir(fullfile(ksroot, 'configFiles', '*.mat'));

idx = 1;
chanMaps = [];
orig_state = warning;
warning('off','all')

for c = 1:numel(chanMapFiles)
    
    q = load(fullfile(ksroot, 'configFiles', chanMapFiles(c).name));
    
    cm = createValidChanMap(q, chanMapFiles(c).name);
    if ~isempty(cm)
        if idx==1
            chanMaps = cm;
        else
            chanMaps = addNewChanMap(chanMaps, cm);
        end
        idx = idx+1;
    end
    
end

warning(orig_state);

end %main function


%% addNewChanMap
function chanMaps = addNewChanMap(chanMaps, cm)
% add new chanMap struct to existing amalgamation of all chanMaps in Kilosort Config dir.
% - bandaid/workaround for convention of holding ALL chan maps in current gui struct at all times
% - previously, constrained chan maps to limited set of fields/info
fnNew = fieldnames(cm);
fnAll = fieldnames(chanMaps);
if all(ismember(fnNew,fnAll)) && length(fnNew)==length(fnAll)
    chanMaps(end+1) = cm;
else
    ii = length(chanMaps)+1;
    for i = 1:length(fnNew)
        chanMaps(ii).(fnNew{i}) = cm.(fnNew{i});
    end
end

end %addNewChanMap
