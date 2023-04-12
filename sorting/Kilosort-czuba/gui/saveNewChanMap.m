function cm = saveNewChanMap(cm, obj)
% function cm = saveNewChanMap(cm, obj)
% 
% [ks25]: updated to ensure unique file name
% - if channel map file already exists, creates unique alternative & polls user for what to do
% - returns [updated] cm struct as output
%
% 2021-06-08  TBC
% 

newName = cm.name;
if strcmp(cm.name, 'unknown')
    answer = inputdlg('Name for this channel map:');
    if ~isempty(answer) && ~isempty(answer{1})
        newName = answer{1};
    else
        obj.log('Cannot save new channel map without a name');
        return
    end
end

% construct new file path
ksRoot = fileparts(fileparts(mfilename('fullpath')));
configDir = fullfile(ksRoot,'configFiles');
newFname = fullfile(configDir, [newName '_kilosortChanMap.mat']);

% Ensure uniqueness
nn = 0;
while exist(newFname, 'file')
    % create unique name
    nn = nn+1;
    newFname = fullfile(configDir, [newName,sprintf('-%02d',nn),'_kilosortChanMap.mat']);
end
if nn
    [~,tmpNew] =fileparts(newFname);
    tmpOld = [newName '_kilosortChanMap'];
    qstr = sprintf('Channel map file "%s" already exists\nA unique alternative was created:\n\t%s\n\nWhat would you like to do?', tmpOld, tmpNew)
    sel = questdlg(qstr, 'Create, overwrite, or skip saving...', 'Use unique', 'Overwrite existing', 'Don''t save', 'Use unique');
    switch lower(sel(1))
        case 'u'
            %update chanmap name field
            cm.name = [newName,sprintf('-%02d',nn)];
        case 'o'
            % overwrite existing
            newFname = fullfile(configDir, [newName '_kilosortChanMap.mat']);
        otherwise
            obj.log('User opted not to save channel map to config dir');
            return
    end
end

% save
save(newFname, '-struct', 'cm');
obj.log(['Saved new channel map:  ' newFname]);

end %main function
