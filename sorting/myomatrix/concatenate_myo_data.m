function concatenate_myo_data(myomatrix_folder, recordings_to_concatenate)
    listing = struct2cell(dir(myomatrix_folder));
    subdir = listing(1, :);
    recordNodeFolders = [];
    dbstop if error
    % Determine Record Node folder
    for i = 1:length(subdir)
        if (contains(subdir(i), 'Record Node'))
            recordNodeFolders = [recordNodeFolders, subdir(i)];
        end
    end
    if length(recordNodeFolders) > 1
        error("Multiple 'Record Node' folders found in the myomatrix folder. Please remove all but one.")
    end
    for i = 1:length(recordNodeFolders)
        currNode = strcat(myomatrix_folder, '/', recordNodeFolders{i});
        folders = struct2cell(dir(currNode));
        subdir = folders(1, :);
        experimentFolders = [];

        % Determine experiment folders
        for j = 1:length(subdir)
            if (startsWith(subdir(j), 'experiment'))
                experimentFolders = [experimentFolders, subdir(j)];
            end
        end
        if length(experimentFolders) > 1
            error("Multiple 'experiment' folders found in the Record Node folder. Please remove all but one.")
        end
        for k = 1:length(experimentFolders)
            currExp = strcat(currNode, '/', experimentFolders{k});
            d = dir(currExp);
            d = d(~ismember({d.name}, {'.', '..', 'concatenated_data'}));
            folders = struct2cell(d);
            subdir = folders(1, :);
            disp("Concatenating recordings: ")
            disp(recordings_to_concatenate{1})
            recordingFolders = [];

            % Determine recording folders
            if class(recordings_to_concatenate{1}) == "char" && recordings_to_concatenate{1} == "all"
                for iRec = 1:length(subdir)
                    if subdir(iRec)==strcat('recording', iRec)
                        recordingFolders = [recordingFolders, subdir(iRec)];
                    end
                end
            elseif class(recordings_to_concatenate{1}) == "double"
                rep_str=repmat('recording',length(recordings_to_concatenate),1);
                recording_str_array = string(cellstr(strcat(rep_str, num2str(recordings_to_concatenate{1}'))));
                % remove any spaces in the middle of the string
                recording_str_array = strrep(recording_str_array, ' ', '');
                for iRec = 1:length(subdir)
                    if ismember(subdir(iRec),recording_str_array)
                        recordingFolders = [recordingFolders, subdir(iRec)];
                    end
                end
            else
                error("Recordings to concatenate must be either 'all' or a double array.")
            end

            continuousFiles = [];
            for m = 1:length(recordingFolders)
                currRecording = strcat(currExp, '/', recordingFolders{m}, '/continuous', '/Acquisition_Board-100.Rhythm Data');
                if ~isfolder(currRecording)
                    currRecording = strcat(currExp, '/', recordingFolders{m}, '/continuous', '/Rhythm_FPGA-100.0');
                    if ~isfolder(currRecording)
                        error('Folder %s does not exist.', currRecording)
                    end
                end
                contFile = strcat(currRecording, '/continuous.dat');
                continuousFiles = [continuousFiles; {contFile}];
            end

            outputDat = [];
            % do concatenation recursively
            if class(continuousFiles) == "cell"
                for n = 1:length(continuousFiles)
                    file = continuousFiles{n};
                    fid = fopen(file, 'r');
                    data = fread(fid, 'int16');
                    outputDat = [outputDat; data];
                    fclose(fid);
                end
            else
                error("Files were not inputted as cell array.")
            end

            % open continuous.dat file for writing
            rhythmFolderNameCellArray = split(currRecording, '/');
            rhythmFolderName = string(rhythmFolderNameCellArray(end)); % get last array element
            concatenated_data_dir = strcat(currExp, '/concatenated_data/', join(string(recordings_to_concatenate{1}),','));
            continuous_folder = strcat(concatenated_data_dir, '/continuous/', rhythmFolderName);
            [~, ~, ~] = mkdir(continuous_folder);
            fid2 = fopen(strcat(continuous_folder, '/continuous.dat'), 'w');
        end

        % write concatenated file
        fwrite(fid2, outputDat, 'int16');
        fclose(fid2);
        % copy a structure.oebin into concatenated_data folder
        lastRecordingFolder = strcat(currExp, '/', recordingFolders{m});
        copyfile(strcat(lastRecordingFolder, '/structure.oebin'), strcat(concatenated_data_dir,'/structure.oebin'));
        disp("Data from " + length(recordingFolders) + " files concatenated together");
    end
    quit
end
