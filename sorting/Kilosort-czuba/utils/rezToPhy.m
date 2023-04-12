function rezToPhy(rez, savePath)
% pull out results from kilosort's rez to save for future analysis,
% and/or manual curation with Phy gui
%
% 2021-03-xx  TBC  Updating to better relect temporal variations in template amplitudes, PCs, etc
% 

%% st3 content:
% % % % From learnAndSolve8b >> runTemplates >> trackAndSort.m
% % %     st3(irange,1) = double(st); % spike times
% % %     st3(irange,2) = double(id0+1); % spike clusters (1-indexing)
% % %     st3(irange,3) = double(x0); % template amplitudes
% % %     st3(irange,4) = double(vexp); % residual variance of this spike
% % %     st3(irange,5) = ibatch; % batch from which this spike was found

%% collect & sort spike vars in local workspace
% spikeTimes will be in samples, not seconds
rez.W = gather(single(rez.Wphy));
rez.U = gather(single(rez.U));
rez.mu = gather(single(rez.mu));


[~, isort]   = sort(rez.st3(:,1), 'ascend');
rez.st3      = rez.st3(isort, :);
rez.cProj    = rez.cProj(isort, :);
rez.cProjPC  = rez.cProjPC(isort, :, :);


fs = dir(fullfile(savePath, '*.npy'));
for i = 1:length(fs)
   delete(fullfile(savePath, fs(i).name));
end
if exist(fullfile(savePath, '.phy'), 'dir')
    rmdir(fullfile(savePath, '.phy'), 's');
end

spikeTimes = uint64(rez.st3(:,1));
% [spikeTimes, ii] = sort(spikeTimes);
spikeTemplates = uint32(rez.st3(:,2));
spikeBatch = uint32(rez.st3(:,5)); 
	

amplitudes = rez.st3(:,3);
% Calc amplitudes to reflect temporal variations in waveform templates


Nchan = rez.ops.Nchan;

xcoords     = rez.xcoords(:);
ycoords     = rez.ycoords(:);
chanMap     = rez.ops.chanMap(:);
chanMap0ind = chanMap - 1;

nt0 = size(rez.W,1);
U = rez.U;
W = rez.W;

Nfilt = size(W,2);

templates = zeros(Nchan, nt0, Nfilt, 'single');
for iNN = 1:size(templates,3)
   templates(:,:,iNN) = squeeze(U(:,iNN,:)) * squeeze(W(:,iNN,:))';
end
templates = permute(templates, [3 2 1]); % now it's nTemplates x nSamples x nChannels
templatesInds = repmat([0:size(templates,3)-1], size(templates,1), 1); % we include all channels so this is trivial

templateFeatures = rez.cProj;
templateFeatureInds = uint32(rez.iNeigh);
pcFeatures = rez.cProjPC;
pcFeatureInds = uint32(rez.iNeighPC);

% whiteningMatrix = rez.Wrot/rez.ops.scaleproc;
whiteningMatrix = eye(size(rez.Wrot)) / rez.ops.scaleproc;
whiteningMatrixInv = whiteningMatrix^-1;

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
tempAmps = zeros(numel(rez.mu),1);
tempAmps(tids) = ta; % because ta only has entries for templates that had at least one spike
gain = getOr(rez.ops, 'gain', 1);
tempAmps = gain*tempAmps'; % for consistency, make first dimension template number

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
        
    for j = 1:length(rez.good)
        if rez.good(j)
            fprintf(fileID, '%d%sgood', j-1, char(9));
        else
            fprintf(fileID, '%d%smua', j-1, char(9));
        end
        fprintf(fileID, char([13 10]));
        
        if isfield(rez, 'est_contam_rate')
            fprintf(fileIDCP, '%d%s%.1f', j-1, char(9), rez.est_contam_rate(j)*100);
            fprintf(fileIDCP, char([13 10]));
        end
        
        fprintf(fileIDA, '%d%s%.1f', j-1, char(9), tempAmps(j));
        fprintf(fileIDA, char([13 10]));
        
    end
    fclose(fileID);
    fclose(fileIDCP);
    fclose(fileIDA);
    
    
    writeNPY(spikeTimes, fullfile(savePath, 'spike_times.npy'));
    writeNPY(uint32(spikeTemplates-1), fullfile(savePath, 'spike_templates.npy')); % -1 for zero indexing
    %if size(rez.st3,2)>4
    %    writeNPY(uint32(spikeClusters-1), fullfile(savePath, 'spike_clusters.npy')); % -1 for zero indexing
    %else
    writeNPY(uint32(spikeTemplates-1), fullfile(savePath, 'spike_clusters.npy')); % -1 for zero indexing
    %end
    writeNPY(amplitudes, fullfile(savePath, 'amplitudes.npy'));
    writeNPY(templates, fullfile(savePath, 'templates.npy'));
    writeNPY(templatesInds, fullfile(savePath, 'templates_ind.npy'));

    %chanMap0ind = int32(chanMap0ind);
    chanMap0ind = int32([1:rez.ops.Nchan]-1);
    writeNPY(chanMap0ind, fullfile(savePath, 'channel_map.npy'));
    writeNPY([xcoords ycoords], fullfile(savePath, 'channel_positions.npy'));

    writeNPY(templateFeatures, fullfile(savePath, 'template_features.npy'));
    writeNPY(templateFeatureInds'-1, fullfile(savePath, 'template_feature_ind.npy'));% -1 for zero indexing
    writeNPY(pcFeatures, fullfile(savePath, 'pc_features.npy'));
    writeNPY(pcFeatureInds'-1, fullfile(savePath, 'pc_feature_ind.npy'));% -1 for zero indexing


    writeNPY(whiteningMatrix, fullfile(savePath, 'whitening_mat.npy'));
    writeNPY(whiteningMatrixInv, fullfile(savePath, 'whitening_mat_inv.npy'));

    if isfield(rez, 'simScore')
        similarTemplates = rez.simScore;
        writeNPY(similarTemplates, fullfile(savePath, 'similar_templates.npy'));
    end

    % Duplicate "KSLabel" as "group", a special metadata ID for Phy, so that
    % filtering works as expected in the cluster view
    KSLabelFilename = fullfile(savePath, 'cluster_KSLabel.tsv');
    copyfile(KSLabelFilename, fullfile(savePath, 'cluster_group.tsv'));

    % % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % % ~~~ No longer necessary/useful ~~~
    % %     driftMap requires complete copy of filtered continuous data ("proDat_....dat")
    % % 
    % % if raw/binary data file location is not same as save destination,
    % % attempt to create symlink to raw file
    % %     if ~strcmpi( fileparts(rez.ops.fbinary), rez.ops.saveDir)
    % %         fprintf(2, ['\n\tWARNING: raw data directory and save output data directory are distinct locations.'...
    % %             '\n\tAttempt to create symlink to raw data in save output directory...']);
    % %         try
    % %             [~, fname, ext] = fileparts(rez.ops.fbinary);
    % %             [err, msg] = system( sprintf('ln -sv %s %s', rez.ops.fbinary, fullfile(rez.ops.saveDir, [fname ext]) ));
    % %             if ~err
    % %                 fprintf('successful!\n\t%s\n',msg)
    % %             else
    % %                 % Note: symlinks won't work on certain file systems (needs extended attributes; not Fat32)
    % %                 fprintf(2, 'failed.\n\t%s',msg)
    % %                 fprintf(['\n\t%s','\n\t>>','\n\t%s'...
    % %                     '\n\tA copy of raw data may need to be added to output directory before starting Phy\n\n'], rez.ops.fbinary,rez.ops.saveDir);
    % %             end
    % %         end
    % %     end
    % % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    %make params file
    if ~exist(fullfile(savePath,'params.py'),'file')
        fid = fopen(fullfile(savePath,'params.py'), 'w');

        % use relative path name for preprocessed data file in params.py
        %     ./proDat_<origRawFileName>.dat 
        [~, fname, ext] = fileparts(rez.ops.fproc);
        fprintf(fid, 'dat_path = ''%s''\n', fullfile('.',[fname,ext]));
        fprintf(fid, 'n_channels_dat = %i\n',rez.ops.Nchan);
        fprintf(fid, 'dtype = ''int16''\n');
        fprintf(fid, 'offset = 0\n');
        if mod(rez.ops.fs,1)
            fprintf(fid,'sample_rate = %i\n',rez.ops.fs);
        else
            fprintf(fid,'sample_rate = %i.\n',rez.ops.fs);
        end
        fprintf(fid, 'hp_filtered = True\n');
        fprintf(fid, 'template_scaling = 15.0\n');
        fclose(fid);
    end
end
