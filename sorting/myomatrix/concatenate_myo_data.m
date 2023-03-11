function concatenate_myo_data(myomatrix_folder)
    listing = struct2cell(dir(myomatrix_folder));
    subdir = listing(1, :);
    recordNodeFiles = [];
    dbstop if error
    % Determine Record Node folder
    for i = 1:length(subdir)
        if (contains(subdir(i), 'Record Node'))
            recordNodeFiles = [recordNodeFiles, subdir(i)];
        end
    end
    for i = 1:length(recordNodeFiles)
        currNode = strcat(myomatrix_folder, '/', recordNodeFiles{i});
        folders = struct2cell(dir(currNode));
        subdir = folders(1, :);
        experimentFiles = [];

        % Determine experiment folders
        for j = 1:length(subdir)
            if (startsWith(subdir(j), 'experiment'))
                experimentFiles = [experimentFiles, subdir(j)];
            end
        end
        for k = 1:length(experimentFiles)
            currExp = strcat(currNode, '/', experimentFiles{k});
            d = dir(currExp);
            d = d(~ismember({d.name}, {'.', '..'}));
            folders = struct2cell(d);
            subdir = folders(1, :);
            disp("Concatenating: ")
            disp(subdir)
            recordingFiles = [];

            % Determine recording folders
            for l = 1:length(subdir)
                if (startsWith(subdir(l), 'recording') && ~contains(subdir(l), '99'))
                    recordingFiles = [recordingFiles, subdir(l)];
                end
            end

            continuousFiles = [];
            for m = 1:length(recordingFiles)
                currRecording = strcat(currExp, '/', recordingFiles{m}, '/continuous', '/Acquisition_Board-100.Rhythm Data');
                if ~isfolder(currRecording)
                    currRecording = strcat(currExp, '/', recordingFiles{m}, '/continuous', '/Rhythm_FPGA-100.0');
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
            concatenated_data_dir = strcat(currExp, '/concatenated_data/');
            continuous_folder = strcat(concatenated_data_dir, 'continuous/', rhythmFolderName);
            [~, ~, ~] = mkdir(continuous_folder);
            fid2 = fopen(strcat(continuous_folder, '/continuous.dat'), 'w');
        end

        % write concatenated file
        fwrite(fid2, outputDat, 'int16');
        fclose(fid2);
        % copy a structure.oebin into concatenated_data folder
        lastRecordingFolder = strcat(currExp, '/', recordingFiles{m});
        copyfile(strcat(lastRecordingFolder, '/structure.oebin'), concatenated_data_dir)
        disp("Data from " + length(recordingFiles) + " files concatenated together");
        quit
    end
end
