function concatenate_myo_data(path, numChannels)
    folder = path;
    listing = struct2cell(dir(folder));
    subdir = listing(1,:);
    recordNodeFiles = [];
    %Determine Record Node Folder
    for i = 1:length(subdir)
        if (contains(subdir(i), 'Record Node'))
            recordNodeFiles = [recordNodeFiles, subdir(i)];
        end
    end
    for i = 1:length(recordNodeFiles)
        currNode = strcat(folder, '/', recordNodeFiles{i});
        folders = struct2cell(dir(currNode));
        subdir = folders(1,:);
        experimentFiles = [];
        %Determine Experiment folder
        for j = 1:length(subdir)
            if (startsWith(subdir(j), 'experiment'))
                experimentFiles = [experimentFiles, subdir(j)];
            end
        end
%         if length(experimentFiles) <= 1
%             disp("Only one recording file found. No concatenation
%             necessary"); quit
%         end
        for k = 1:length(experimentFiles)
            currExp = strcat(currNode, '/', experimentFiles{k});
            folders = struct2cell(dir(currExp));
            subdir = folders(1,:);
            %disp(subdir)
            recordingFiles = [];
            %Determine recording folder
            for l = 1:length(subdir)
                if (startsWith(subdir(l), 'recording') && ~contains(subdir(l),'99'))
                    recordingFiles = [recordingFiles, subdir(l)];
                end
            end
            continuousFiles = [];
            for m = 1:length(recordingFiles)
                currRecording = strcat(currExp, '/', recordingFiles{m}, '/continuous', '/Acquisition_Board-100.Rhythm Data');
                if ~isfolder(currRecording)
                    currRecording = strcat(currExp, '/', recordingFiles{m}, '/continuous', '/Rhythm_FPGA-100.0');
                    if ~isfolder(currRecording)
                        error('Folder %s does not exist.',currRecording)
                    end
                end
                contFile = strcat(currRecording, '/continuous.dat');
                continuousFiles = [continuousFiles; {contFile}];
            end
            %continuousFiles = cellstr(continuousFiles);
            outputDat = [];
            if class(continuousFiles) == "cell"
                for n = 1:length(continuousFiles)
                    file = continuousFiles{n};
                    fid = fopen(file,'r');
                    data = fread(fid, 'int16');
                    %disp(length(data));
                    outputDat = [outputDat; data];
                    fclose(fid);
                end
            else
                error("Files were not inputted as cell array.")
            end
            rhythmFolderNameCellArray = split(currRecording,'/');
            rhythmFolderName = string(rhythmFolderNameCellArray(end)); % get last array element
            concatFile = strcat(currExp, '/concatenated_data/');
            continuous_folder = strcat(concatFile, 'continuous/', rhythmFolderName);
            [~, ~, ~] = mkdir(continuous_folder);
            fid2 = fopen(strcat(continuous_folder,'/continuous.dat'), 'w');
            %num_chans = input("what is the number of channels?","s");
            %num_chans = str2double(num_chans);
            num_chans = numChannels;
            dataMat = reshape(outputDat,num_chans,length(outputDat)/num_chans);
            testing = 1;
            if testing
                fs = 30000;
                y1 = fft(dataMat);
                N = length(dataMat);          % number of samples
                f = (0:N-1)*(fs/N);     % frequency range
                pow = abs(y1).^2/N;    % power of the DFT
                plot(f,pow), title('Before bandpass filtering');
            end
            [b_pass,a_pass] = butter(4, [300,7000]/(30000/2), 'bandpass');
            %keyboard
            filtDataMat = filtfilt(b_pass, a_pass, dataMat');
            if testing   
                fs = 30000;
                y2 = fft(filtDataMat);
                N = length(filtDataMat);          % number of samples
                f = (0:N-1)*(fs/N);     % frequency range
                pow = abs(y2).^2/N;    % power of the DFT
                plot(f,pow), title('After bandpass filtering');
            end
            %             [filtDataMat, order] = bandpass(dataMat',[300,7500],30000);
            %keyboard
            outputDat = reshape(filtDataMat',length(outputDat),1);
            fwrite(fid2, outputDat, 'int16')
            fclose(fid2);
            %disp(length(outputDat));
            %[~, ~, ~] = mkdir(strcat(concatFile,'/KS_rez'));
            % copy a structure.oebin into recording99 folder
            lastRecordingFolder = strcat(currExp, '/', recordingFiles{m});
            copyfile(strcat(lastRecordingFolder,'/structure.oebin'),concatFile)
        end
    end
    disp("Data from " + length(recordingFiles) + " files concatenated together");
    quit
end