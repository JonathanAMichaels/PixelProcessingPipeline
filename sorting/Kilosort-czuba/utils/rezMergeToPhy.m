function rezMergeToPhy(rez1, rez2, savePath)
% function rezMergeToPhy(rez1, rez2, savePath)
% 
% Merge two kilosort rez structs into one,
% Save all requisit output files for loading merged dataset into Phy
%
%   ~~~ Not Recommended ~~~
%   Integration of template & feature projections of two independent
%   kilosort sessions requires recomputing all spike projections
%   and reassessing template similarity across merged set.
%   ...short of that, only thing recoverable from rough merge of two
%   sessions is really high amp units that would probably be tracked
%   just fine if they were sorted together in the first place.
% 
% ---
%   W.I.P.:: does not handle template or feature projections, which
%   are typically excluded from rez.mat save struct, and should really
%   be recomputed based on merged content (e.g. template similarity 
%   & feature projections of coherent clusters from each rez session)
% ---
% 2021-06-xx  TBC  Hacked together based on standard rezToPhy.m
% 2021-06-21  TBC  Abandon all hope, ye who enter here...
% 


%% st3 content:
% % % % From learnAndSolve8b >> runTemplates >> trackAndSort.m
% % %     st3(irange,1) = double(st); % spike times
% % %     st3(irange,2) = double(id0+1); % spike clusters (1-indexing)
% % %     st3(irange,3) = double(x0); % template amplitudes
% % %     st3(irange,4) = double(vexp); % residual variance of this spike
% % %     st3(irange,5) = ibatch; % batch from which this spike was found


%% Parse inputs & combine rez structs

if ~exist(savePath,'dir')
    mkdir(savePath);
elseif strcmp(rez1.ops.saveDir, savePath) || strcmp(rez2.ops.saveDir, savePath)
    error('Merged destination directory cannot be the same as either of the input rez structs.');
elseif exist(savePath,'dir')
    savePath = uigetdir(savePath, 'Destination exists, please confirm');
end
    
    
% add index for each rez struct
rez1.rid = 1;
rez2.rid = 2;

% combine rez structs
rez(1) = rez1;
rez(2) = rez2;
ntemps = cumsum([0, arrayfun(@(x) length(x.mu), rez)]);


%% clear input rez vars (excess memory overhead)
clear rez1 rez2


%% clear existing/conflicting files from destination
fs = dir(fullfile(savePath, '*.npy'));
for i = 1:length(fs)
   delete(fullfile(savePath, fs(i).name));
end
if exist(fullfile(savePath, '.phy'), 'dir')
    rmdir(fullfile(savePath, '.phy'), 's');
end


%% Compile params from input rez structs
spikeTimes = cell2mat(arrayfun(@(x) uint64(x.st3(:,1)), rez, 'uni',0)');
% spikeTimes = cell2mat(spikeTimes'); % concatenate

[spikeTimes, ii] = sort(spikeTimes);

% - add offset to template indices of second rez struct to ensure ids are unique
% - offset must match with index of concatenated template shapes as well
spikeTemplates = cell2mat(arrayfun(@(x) uint32(x.st3(:,2) + ntemps(x.rid)), rez, 'uni',0)');
spikeTemplates = spikeTemplates(ii);
% NO:  st3(:,5) is really batch#, not cluster# (!??...KS1 holdover?)
% if size(rez.st3,2)>4
%     spikeClusters = uint32(1+rez.st3(:,5));
% end

... unused:     spikeBatch = uint32(rez.st3(:,5)); 

amplitudes = cell2mat(arrayfun(@(x) x.st3(:,3), rez, 'uni',0)');
amplitudes = amplitudes(ii);
% Calc amplitudes to reflect temporal variations in waveform templates
isgood = cell2mat(arrayfun(@(x) x.good, rez, 'uni',0)');

estContam = cell2mat(arrayfun(@(x) x.est_contam_rate, rez, 'uni',0)');

% the following fields MUST BE IDENTICAL for both rez structs
Nchan = rez(1).ops.Nchan;

xcoords     = rez(1).xcoords(:);
ycoords     = rez(1).ycoords(:);
chanMap     = rez(1).ops.chanMap(:);
chanMap0ind = chanMap - 1;

nt0 = size(rez(1).W,1);


U = arrayfun(@(x) x.U, rez, 'uni',0);
U = cat(2, U{:}); % must do two step for multi dimensional
W = arrayfun(@(x) x.W, rez, 'uni',0);
W = cat(2, W{:});

% total number of templates
Nfilt = ntemps(end);% size(W,2);

templates = zeros(Nchan, nt0, Nfilt, 'single');
for iNN = 1:size(templates,3)
   templates(:,:,iNN) = squeeze(U(:,iNN,:)) * squeeze(W(:,iNN,:))';
end
templates = permute(templates, [3 2 1]); % now it's nTemplates x nSamples x nChannels
templatesInds = repmat((0:size(templates,3)-1), size(templates,1), 1); % we include all channels so this is trivial

%% Feature & PC projections
% Nope, this really fails. Kludge between this half-measure and Phy's readout of these features
% is no better than actually concatenating the two Kilosort sessions in the first place
% 
%         % cProj & cProjPC fields are typically excluded from rez.mat save because can balloon file into gigs of data
%         % - simply concatenating these values is not quite legitimate, but may roughly gets the job done
%         %   (...with no more evils than already present in standard kilosort feature calc)
%         % - Really, this should recompute features, simiilarity, & pc projections based on concatenated template set (W & U).
%         %   BUT that involves running through every template & spike waveform (extracted from processed data)
%         %   which would really need it's own CUDA function, but is probably better done w/in Phy anyway....
%         % - ...so this will have to do for now.
%         if isfield(rez, 'cProj') && all(arrayfun(@(x) ~isempty(x.cProj), rez))
%             % cProj are template feature projections 
%             templateFeatures = cell2mat(arrayfun(@(x) x.cProj, rez, 'uni',0)');
%             % iNeigh are indices into similar **templates** & need to be adjusted to match concatenated template indices
%             templateFeatureInds = arrayfun(@(x) uint32(x.iNeigh + ntemps(x.rid)), rez, 'uni',0);
%             templateFeatureInds = cat(2, templateFeatureInds{:});
%             % cProjPC are PC projections onto nearby channels
%             pcFeatures = arrayfun(@(x) x.cProjPC, rez, 'uni',0);
%             pcFeatures = cat(1, pcFeatures{:});
%             % iNeighPC are indices into nearby **channels** & DO NOT need to be adjusted
%             pcFeatureInds = arrayfun(@(x) uint32(x.iNeighPC), rez, 'uni',0);
%             pcFeatureInds = cat(2, pcFeatureInds{:});
% 
%             %     templateFeatures = rez.cProj;
%             %     templateFeatureInds = uint32(rez.iNeigh);
%             %     pcFeatures = rez.cProjPC;
%             %     pcFeatureInds = uint32(rez.iNeighPC);
%         end

% Combine whitening matrix & inverse
% Here things get tricky...or maybe not.
% - rezToPhy stopped using the actual whitening matrix when transitioned to datashift method;
%   'whitening_mat.npy' (& the inverse) is just undoing the scaleproc now.
%           whiteningMatrix = rez.Wrot/rez.ops.scaleproc;               % pre-datashift
%           whiteningMatrix = eye(size(rez.Wrot)) / rez.ops.scaleproc;  % post-datashift
% So as long as both structs use the same scaleproc, this should be fine
if rez(1).ops.scaleproc ~= rez(end).ops.scaleproc
    warning('Incompatible scaling parameters used in rez structs. [rez.ops.scaleproc] must be identical.')
    keyboard
end
whiteningMatrix = eye(size(rez(1).Wrot)) / rez(1).ops.scaleproc;
whiteningMatrixInv = whiteningMatrix^-1;


%% This section should all 'just work' on the concatenated data
% here we compute the amplitude of every template...

% unwhiten all the templates
tempsUnW = zeros(size(templates));
for t = 1:size(templates,1)
    tempsUnW(t,:,:) = squeeze(templates(t,:,:))*whiteningMatrixInv;
end

% The amplitude on each channel is the positive peak minus the negative
tempChanAmps = squeeze(max(tempsUnW,[],2))-squeeze(min(tempsUnW,[],2));

% The template amplitude is the amplitude of its largest channel
tempAmpsUnscaled = max(tempChanAmps,[],2);

% assign all spikes the amplitude of their template multiplied by their
% scaling amplitudes
spikeAmps = tempAmpsUnscaled(spikeTemplates).*amplitudes;

% take the average of all spike amps to get actual template amps (since
% tempScalingAmps are equal mean for all templates)
ta = clusterAverage(spikeTemplates, spikeAmps);
tids = unique(spikeTemplates);
tempAmps = zeros(ntemps(end),1);       % zeros(numel(rez.mu),1);
tempAmps(tids) = ta; % because ta only has entries for templates that had at least one spike
tempAmps = tempAmps';   % gain is fixed
%     gain = getOr(rez.ops, 'gain', 1);
%     tempAmps = gain*tempAmps'; % for consistency, make first dimension template number

if ~isempty(savePath)
    fileID = fopen(fullfile(savePath, 'cluster_KSLabel.tsv'),'w');
    fprintf(fileID, 'cluster_id%sKSLabel', char(9));
    fprintf(fileID, char([13 10]));
    
    fileIDCP = fopen(fullfile(savePath, 'cluster_ContamPct.tsv'),'w');
    fprintf(fileIDCP, 'cluster_id%sContamPct', char(9));
    fprintf(fileIDCP, char([13 10]));
    
    fileIDA = fopen(fullfile(savePath, 'cluster_Amplitude.tsv'),'w');
    fprintf(fileIDA, 'cluster_id%sAmplitude', char(9));
    fprintf(fileIDA, char([13 10]));
        
    for j = 1:length(isgood)
        if isgood(j)
            fprintf(fileID, '%d%sgood', j-1, char(9));
        else
            fprintf(fileID, '%d%smua', j-1, char(9));
        end
        fprintf(fileID, char([13 10]));
        
        if isfield(rez, 'est_contam_rate')
            fprintf(fileIDCP, '%d%s%.1f', j-1, char(9), estContam(j)*100);
            fprintf(fileIDCP, char([13 10]));
        end
        
        fprintf(fileIDA, '%d%s%.1f', j-1, char(9), tempAmps(j));
        fprintf(fileIDA, char([13 10]));
        
    end
    fclose(fileID);
    fclose(fileIDCP);
    fclose(fileIDA);
    
    
    writeNPY(spikeTimes,    fullfile(savePath, 'spike_times.npy'));
    writeNPY(uint32(spikeTemplates-1),  fullfile(savePath, 'spike_templates.npy')); % -1 for zero indexing

    writeNPY(amplitudes,    fullfile(savePath, 'amplitudes.npy'));
    writeNPY(templates,     fullfile(savePath, 'templates.npy'));
    writeNPY(templatesInds, fullfile(savePath, 'templates_ind.npy'));

    %chanMap0ind = int32(chanMap0ind);
    chanMap0ind = int32([1:Nchan]-1);
    writeNPY(chanMap0ind,   fullfile(savePath, 'channel_map.npy'));
    writeNPY([xcoords ycoords],     fullfile(savePath, 'channel_positions.npy'));
    
    % % Template projections may be salvagable, but exclude for now
    %     if exist('templateFeatures','var')
    %         writeNPY(templateFeatures, fullfile(savePath, 'template_features.npy'));
    %         writeNPY(templateFeatureInds'-1, fullfile(savePath, 'template_feature_ind.npy'));% -1 for zero indexing
    %     end
    
    % % Feature projections excluded from rez merge...must be fully recomputed, & beyond scope of this bandaid
    %     if exist('pcFeatures','var')    
    %         writeNPY(pcFeatures, fullfile(savePath, 'pc_features.npy'));
    %         writeNPY(pcFeatureInds'-1, fullfile(savePath, 'pc_feature_ind.npy'));% -1 for zero indexing
    %     end

    writeNPY(whiteningMatrix,       fullfile(savePath, 'whitening_mat.npy'));
    writeNPY(whiteningMatrixInv,    fullfile(savePath, 'whitening_mat_inv.npy'));

    if isfield(rez, 'simScore')
        % similarTemplates = cell2mat(arrayfun(@(x) x.simScore, rez, 'uni',0)');
        similarTemplates = zeros(ntemps(end));
        sims = arrayfun(@(x) x.simScore, rez, 'uni',0);
        for i = 1:length(sims)
            nt = size(sims{i},1);
            ii = (1:nt)+ntemps(i);
            similarTemplates(ii,ii) = sims{i};
        end
        writeNPY(similarTemplates, fullfile(savePath, 'similar_templates.npy'));
    end


    % Duplicate "KSLabel" as "group", a special metadata ID for Phy, so that
    % filtering works as expected in the cluster view
    KSLabelFilename = fullfile(savePath, 'cluster_KSLabel.tsv');
    copyfile(KSLabelFilename, fullfile(savePath, 'cluster_group.tsv'));

    %make params file
    if ~exist(fullfile(savePath,'params.py'),'file')
        fid = fopen(fullfile(savePath,'params.py'), 'w');

        % use relative path name for preprocessed data file in params.py
        % - defaults to preprocessed file of last rez struct
        % - assuming they're in order
        % - **** AND that the preprocessed data file was created with the [ks25] branch
        %   - which include everything in the preprocessed file from t0 to tend
        [~, fname, ext] = fileparts(rez(end).ops.fproc);
        copyfile(rez(end).ops.fproc, fullfile(savePath, [fname,ext]));
        fprintf(fid, 'dat_path = ''%s''\n', fullfile('.',[fname,ext]));
        fprintf(fid,'n_channels_dat = %i\n', Nchan);
        fprintf(fid,'dtype = ''int16''\n');
        fprintf(fid,'offset = 0\n');
        if mod(rez(1).ops.fs,1)
            fprintf(fid,'sample_rate = %i\n', rez(1).ops.fs);
        else
            fprintf(fid,'sample_rate = %i.\n', rez(1).ops.fs);
        end
        fprintf(fid,'hp_filtered = True\n');
        fprintf(fid,'template_scaling = 5.0\n');
        
        fclose(fid);
    end
end

end %main function

