

function cm = createValidChanMap(q, varargin)
cm = [];
if isfield(q, 'chanMap') && ...
        isfield(q, 'xcoords') && ...
        isfield(q, 'ycoords') && ...
        numel(q.chanMap)==numel(q.xcoords) &&...
        numel(q.chanMap)==numel(q.ycoords)
    
    if min(q.chanMap)==0
        q.chanMap = q.chanMap+1;
    end
        
    % has required fields - we accept it
    
    cm.chanMap = q.chanMap(:);
    cm.xcoords = q.xcoords(:);
    cm.ycoords = q.ycoords(:);
    cm.chanMap0ind = q.chanMap(:)-1;   
    
    % [.connected] logical flag for connected channels
    % - utility unclear...if not connected, why present in chanMap??
    if isfield(q, 'connected') 
        if numel(q.connected)==numel(cm.chanMap) 
            cm.connected = q.connected(:);
        else
            warning('Invalid chanMap variable ''connected'': must be the same size as chanMap. Using default.');
        end
        
    else
        cm.connected = true(size(cm.chanMap));
    end
    
    % [.fs] continuous data sampling rate
    if isfield(q, 'fs') 
        cm.fs = q.fs;
    else
        cm.fs = [];
    end
    
    % [.kcoords] indexed grouping of separate devices/shanks w/in same recording
    % - not used since Kilosort 2.0
    cm.kcoords = ones(size(cm.chanMap));
    if isfield(q, 'kcoords') 
        if numel(q.kcoords)==numel(cm.chanMap)
            cm.kcoords = q.kcoords(:);    
        else
            warning('Invalid chanMap variable ''kcoords'': must be the same size as chanMap. Using default.');
        end
    elseif isfield(q, 'shankInd')
        if numel(q.shankInd)==numel(cm.chanMap)           
            cm.kcoords = q.shankInd(:);        
        else
            warning('Invalid chanMap variable ''shankInd'': must be the same size as chanMap. Using default.');
        end            
    end
    
    if isfield(q, 'name')
        cm.name = q.name;
    else
        if nargin>1
            filename = varargin{1};
            x = strfind(filename, 'kilosortChanMap');
            if ~isempty(x)
                cm.name = filename(1:x-2);
            else
                cm.name = filename;
            end
        else
            cm.name = 'unknown';
        end
    end
    
    if isfield(q, 'siteSize')
        cm.siteSize = q.siteSize;
    end
    
    % Don't discard additional chanMap fields
    % - allow user to include additional info about their recording/hardware/etc
    fn = fieldnames(q);
    for f = find(~ismember(fn, fieldnames(cm)))'
        cm.(fn{f}) = q.(fn{f});
    end
    
else
    warning('Invalid channel map: A valid channel map must have chanMap, xcoords, and ycoords, and they must all be the same size.')
end