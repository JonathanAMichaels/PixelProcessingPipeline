classdef ksGUI < handle
    % GUI for kilosort
    %
    % Purpose is to display some basic data and diagnostics, to make it easy to
    % run kilosort
    %
    % Kilosort by M. Pachitariu
    % GUI by N. Steinmetz
    %
    % TODO: (* = before release)
    % - allow better setting of probe site shape/size
    % - auto-load number of channels from meta file when possible
    % - update time plot when scrolling in dataview
    % - show RMS noise level of channels to help selecting ones to drop?
    % - implement builder for new probe channel maps (cm, xc, yc, name,
    % site size)
    % - saving of probe layouts
    % - plotting bug: zoom out on probe view should allow all the way out
    % in x
    % - some help/tools for working with other datafile types
    % - update data view needs refactoring... load a bigger-than-needed
    % segment of data, and just move around using this as possible
    % - when re-loading, check whether preprocessing can be skipped 
    % - find way to run ks in the background so gui is still usable(?)
    % - quick way to set working/output directory when selecting a new file
    % - when selecting a new file, reset everything
    % - why doesn't computeWhitening run on initial load?
    % 
    % - disable automatic save settings on close (TBC)

    properties        
        H % struct of handles to useful parts of the gui
        
        P % struct of parameters of the gui to remember, like last working directory        
        
        ops % struct for kilosort to run
        
        rez % struct of results of running
        
    end
    
    methods
        function obj = ksGUI(parent)
            
            obj.init();
            
            obj.build(parent);
            
            obj.initPars(); % happens after build since graphical elements are needed
            
            % bring kilosort figure to focus
            figure(obj.H.fig)
        end
        
        function init(obj)
            
            % check that required functions are present
            if ~exist('uiextras.HBox','class')
                error('ksGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Home->Add-Ons->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson. You may have to search for the author''s name to find the right one for some reason. If you cannot find it, go here to download: https://www.mathworks.com/matlabcentral/fileexchange/47982-gui-layout-toolbox\n')
            end
            
            % add paths
            mfPath = mfilename('fullpath');            
            if ~exist('readNPY','file')
                githubDir = fileparts(fileparts(fileparts(mfPath))); % taking a guess that they have a directory with all github repos
                if exist(fullfile(githubDir, 'npy-matlab'))
                    addpath(genpath(fullfile(githubDir, 'npy-matlab')));
                end
            end
            if ~exist('readNPY','file')
                warning('ksGUI:init:npy', 'In order to save data for phy, you must have the npy-matlab repository from https://github.com/kwikteam/npy-matlab in your matlab path\n');
            end
            
            % compile if necessary
            if exist('mexSVDsmall2','file')~=3
                
                fprintf(1, 'Compiled Kilosort files not found. Attempting to compile now.\n');
                try
                    oldDir = pwd;
                    cd(fullfile(fileparts(fileparts(mfPath)), 'CUDA'));
                    mexGPUall;
                    fprintf(1, 'Success!\n');
                    cd(oldDir);
                catch ex
                    fprintf(1, 'Compilation failed. Check installation instructions at https://github.com/MouseLand/Kilosort2\n');
                    rethrow(ex);
                end
            end
            
            obj.P.allChanMaps = loadChanMaps();         
                        
        end
        
        
        function build(obj, f)
            % construct the GUI with appropriate panels
            obj.H.fig = f;
            obj.H.fsz = get(groot, 'defaultUicontrolFontSize'); % base font size  ['defaultUicontrolFontSize' OR 'defaultAxesFontSize']
            obj.H.tracelw = 1; % base trace linewidth
            set(f, 'UserData', obj);
            
            set(f, 'KeyPressFcn', @(f,k)obj.keyboardFcn(f, k));
            
            obj.H.root = uiextras.VBox('Parent', f,...
                'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
                'Padding', 5);
            
            % - Root sections
            obj.H.titleHBox = uiextras.HBox('Parent', obj.H.root, 'Spacing', 50);                        
            
            obj.H.mainSection = uiextras.HBox(...
                'Parent', obj.H.root);
            
%             obj.H.logPanel = uiextras.Panel(...
%                 'Parent', obj.H.root, ...
%                 'Title', 'Message Log', 'FontSize', 1*obj.H.fsz,...
%                 'FontName', 'Myriad Pro');
            
            obj.H.root.Sizes = [-1 -20];    %[-1 -20 -3];
            
            % -- Title bar
            obj.H.titleBar = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'text', 'HorizontalAlignment', 'left', ...
                'String', 'Kilosort', 'FontSize', 2*obj.H.fsz,...
                'FontName', 'Myriad Pro', 'FontWeight', 'bold');
            
            obj.H.helpButton = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'pushbutton', ...
                'String', 'Help', 'FontSize', 1.5*obj.H.fsz,...
                'Callback', @(~,~)obj.help);
            
            obj.H.resetButton = uicontrol(...
                'Parent', obj.H.titleHBox,...
                'Style', 'pushbutton', ...
                'String', 'Reset GUI', 'FontSize', 1.5*obj.H.fsz,...
                'Callback', @(~,~)obj.reset);
            
            obj.H.titleHBox.Sizes = [-8 -1 -1];
            
            % -- Main section
            obj.H.setRunVBox = uiextras.VBox(...
                'Parent', obj.H.mainSection);
            
            obj.H.settingsPanel = uiextras.Panel(...
                'Parent', obj.H.setRunVBox, ...
                'Title', 'Settings', 'FontSize', 1.2*obj.H.fsz,...
                'FontName', 'Myriad Pro');
            obj.H.runPanel = uiextras.Panel(...
                'Parent', obj.H.setRunVBox, ...
                'Title', 'Run', 'FontSize', 1.2*obj.H.fsz,...
                'FontName', 'Myriad Pro');

            obj.H.logPanel = uiextras.Panel(...
                'Parent', obj.H.setRunVBox, ... obj.H.root, ...
                'Title', 'Message Log', 'FontSize', 1*obj.H.fsz,...
                'FontName', 'Myriad Pro');
            
            obj.H.setRunVBox.Sizes = [-8 -2 -1];    %[-4 -1];
            
            obj.H.probePanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Probe view', 'FontSize', 1.2*obj.H.fsz,...
                'FontName', 'Myriad Pro', 'Padding', 3*obj.H.fsz);
            
            obj.H.dataPanel = uiextras.Panel(...
                'Parent', obj.H.mainSection, ...
                'Title', 'Data view', 'FontSize', 1.2*obj.H.fsz,...
                'FontName', 'Myriad Pro', 'Padding', 5);
            
            obj.H.mainSection.Sizes = [-3 -1 -7];

            
            %% Create Settings panel
            obj.H.settingsVBox = uiextras.VBox(...
                'Parent', obj.H.settingsPanel);
            
            obj.H.settingsGrid = uiextras.Grid(...
                'Parent', obj.H.settingsVBox, ...
                'Spacing', 10, 'Padding', 5);
            
            % ---Settings Panel, Left column-------------------------------------------
            % ---Labels
            
            % Raw data file selection button
            obj.H.settings.ChooseFileTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Select data file', ...
                'Callback', @(~,~)obj.selectFileDlg);
                        
            % Results output directory selection button
            obj.H.settings.ChooseOutputTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', '<html><center>Select results<br/>output directory</center></html>', ... sprintf('Select results \noutput directory'), ...
                'Callback', @(s,~)obj.selectDirDlg(s),...
                'Tag', 'output');
                        
            % choose probe
            obj.H.settings.setProbeTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Select probe layout');
            
            % set nChannels
            obj.H.settings.setnChanTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Number of channels');
                        
            % set Fs
            obj.H.settings.setFsTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Sampling frequency (Hz)');
                        
            % set time range
            obj.H.settings.setTrangeTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Time range (s)');
            
            % Drift correction scale/mode
            % - coopted from "good channels" min firing rate (krufty hack inherited from main repo)
            obj.H.settings.setMinfrTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', {'Drift Correction ("nblocks")', '(0=none, 1=rigid, N=nonrigid)'});
            
            % Cluster template threshold
            obj.H.settings.setThTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Threshold');

            % set lambda
            obj.H.settings.setLambdaTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'Lambda');
            
            % set AUC split/merge threshold
            % - deprecated in main Kilosort (v3.0) repo, BUT NOT in [ks25] (>>splitAllClusters.m)
            obj.H.settings.setCcsplitTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'text', 'HorizontalAlignment', 'right', ...
                'String', 'AUC for split/merge');
            
            % [Empty] place filler for ui settings grid
            uiextras.Empty('Parent', obj.H.settingsGrid);
            
            nSettingsRows = numel(get(obj.H.settingsGrid,'Children'));
            
            % ---Settings Panel, Right column-------------------------------------------
            % ---Parameter values/text boxes
            
            % Raw data file  (--> ops.fbinary)
            obj.H.settings.ChooseFileEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...', 'Callback', @(~,~)obj.updateFileSettings());
            
            % Results output directory
            obj.H.settings.ChooseOutputEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'String', '...', 'Callback', @(~,~)obj.updateFileSettings());
            
            % Probe channel map dropdown menu
            probeNames = {obj.P.allChanMaps.name};
            probeNames{end+1} = '[new]'; 
            probeNames{end+1} = 'other...'; 
            obj.H.settings.setProbeEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'popupmenu', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', probeNames, ...
                'Callback', @(~,~)obj.updateProbeView('reset'));
            
            % Total channel count 
            obj.H.settings.setnChanEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '', 'Callback', @(~,~)obj.updateFileSettings());
            % Sampling frequency (Hz)
            obj.H.settings.setFsEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '40000', 'Callback', @(~,~)obj.updateFileSettings());
            % Time range (s)
            obj.H.settings.setTrangeEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '', 'Callback', @(~,~)obj.updateFileSettings());
            % Drift Correction scale
            % - coopted from min firing rate (krufty hack inherited from main repo)
            obj.H.settings.setMinfrEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '');
            % Threshold
            obj.H.settings.setThEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '');
            % Lambda
            obj.H.settings.setLambdaEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '');
            % AUC for split/merge
            obj.H.settings.setCcsplitEdt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'edit', 'HorizontalAlignment', 'left', ...
                'FontSize', 0.8*obj.H.fsz,...
                'String', '');
            
            % Advanced options button
            obj.H.settings.setAdvancedTxt = uicontrol(...
                'Parent', obj.H.settingsGrid,...
                'Style', 'pushbutton', ...
                'String', 'Set advanced options', ...
                'FontSize', 0.8*obj.H.fsz,...
                'Callback', @(~,~)obj.advancedPopup());
            
            set(obj.H.settingsGrid, 'ColumnSizes',[-1 -3], ...
                'RowSizes',[-1*ones(1,nSettingsRows-1),obj.H.fsz*2.5]); % [nSettingsRows] est. after left column codeblock

            %% Create Run button panel            
            obj.H.runVBox = uiextras.VBox(...
                'Parent', obj.H.runPanel,...
                'Spacing', 5, 'Padding', 5);
            
            obj.H.runHBox = uiextras.HBox(...
                'Parent', obj.H.runVBox,...
                'Spacing', 5, 'Padding', 5);
            
            % Left column:  Run All button
            obj.H.settings.runBtn = uicontrol(...
                'Parent', obj.H.runHBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Run All', 'enable', 'off', ...
                'FontSize', 1.4*obj.H.fsz,...
                'Callback', @(~,~)obj.runAll());
            
            % Middle column
            obj.H.settings.runEachVBox = uiextras.VBox(...
                'Parent', obj.H.runHBox,...
                'Spacing', 3, 'Padding', 3);

            % Right column
            obj.H.settings.runSortSaveVBox = uiextras.VBox(...
                'Parent', obj.H.runHBox,...
                'Spacing', 3, 'Padding', 3);
            
            obj.H.runHBox.Sizes = [-1 -2 -1];
            
            % Preprocess button
            obj.H.settings.runPreprocBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Preprocess', 'enable', 'off', ...
                'FontSize', 1.2*obj.H.fsz,...
                'Callback', @(~,~)obj.runPreproc());
            
            % Spikesort button
            obj.H.settings.runSpikesortBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Spikesort', 'enable', 'off', ...
                'FontSize', 1.2*obj.H.fsz,...
                'Callback', @(~,~)obj.runSpikesort());
            
            % Save for Phy button
            obj.H.settings.runSaveBtn = uicontrol(...
                'Parent', obj.H.settings.runEachVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Save for Phy', 'enable', 'off', ...
                'FontSize', 1.2*obj.H.fsz,...
                'Callback', @(~,~)obj.runSaveToPhy());
            
            %     % Save defaults button
            %     obj.H.settings.saveBtn = uicontrol(...
            %         'Parent', obj.H.runVBox,...
            %         'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
            %         'String', 'Save state', ...
            %         'FontSize', 1.2*obj.H.fsz,...
            %         'Callback', @(~,~)obj.saveGUIsettings());
            
            % Sort & Save button
            uiextras.Empty('Parent', obj.H.settings.runSortSaveVBox);
            obj.H.settings.runSortSaveBtn = uicontrol(...
                'Parent', obj.H.settings.runSortSaveVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', '<html><center>Sort &<br/>Save</center></html>', 'enable', 'off', ...
                'FontSize', 1.2*obj.H.fsz,...
                'Callback', @(~,~)obj.runSortAndSave());
            obj.H.settings.runSortSaveVBox.Sizes = [-1,-2];
            
            obj.H.runVBox.Sizes = [-1];
            
            
            %% Create Probe View panel
            obj.H.probeAx = axes(obj.H.probePanel, 'ActivePositionProperty', 'Position');
            set(obj.H.probeAx, 'ButtonDownFcn', @(f,k)obj.probeClickCB(f, k),...
                'color','none', 'xcolor',.4*[1 1 1], 'ycolor',.4*[1 1 1], 'box','on', ...
                'xtick',[], 'ytick',[]);
            hold(obj.H.probeAx, 'on');            
            
            
            %% Create Data View panel
            obj.H.dataVBox = uiextras.VBox('Parent', ...
                obj.H.dataPanel, 'Padding', 50, 'Spacing', 50);
            
            obj.H.dataControlsTxt = uicontrol('Parent', obj.H.dataVBox,...
                'Style', 'pushbutton', 'HorizontalAlignment', 'left', ...
                'String', 'Controls',...
                'FontWeight', 'bold', ...
                'Callback', @(~,~)helpdlg({'Controls:','','--------','',...
                ' [1, 2, 3, 4] Enable different data views','',...
                ' [c] Toggle colormap vs traces mode', '',...
                ' [up, down] Add/remove channels to be displayed', '',...
                ' [scroll and alt/ctrl/shift+scroll] Zoom/scale/move ', '',...
                ' [click] Jump to position and time', '',...
                ' [right click] Disable nearest channel'}));
            
            obj.H.dataAx = axes(obj.H.dataVBox, 'ActivePositionProperty', 'Position');   
            
            set(obj.H.dataAx, 'ButtonDownFcn', @(f,k)obj.dataClickCB(f, k));
            set(obj.H.dataAx, 'TickLength',[.005, .005])
            title(obj.H.dataAx,'');
            % initialize data trace handles  [raw, filtered, predicted, residual]
            [obj.H.residTr,  obj.H.predTr, obj.H.ppTr, obj.H.dataTr] = deal([]);
            hold(obj.H.probeAx, 'on');
            
            set(obj.H.fig, 'WindowScrollWheelFcn', @(src,evt)obj.scrollCB(src,evt))
            set(obj.H.fig, 'WindowButtonMotionFcn', @(src, evt)any(1));
            
            
            %% Create Data View - Timeline slider
            obj.H.timeAx = axes(obj.H.dataVBox);
            set(obj.H.timeAx, 'NextPlot','add');
            sq = [0 0; 0 1; 1 1; 1 0];
            stepW = 0.02; % time step arrow width
            
            % step left (backward)
            obj.H.timeLBtn = fill(obj.H.timeAx, [sq(1:2,1)-stepW/4;-stepW], [sq(1:2,2);0.5], 0.65*[1 1 1]);
            % step right (forward)
            obj.H.timeRBtn = fill(obj.H.timeAx, [sq(3:4,1)+stepW/4;1+stepW], [sq(3:4,2);0.5], 0.65*[1 1 1]);
            % time axis background
            obj.H.timeBckg = fill(obj.H.timeAx, sq(:,1), sq(:,2), [0.3 0.3 0.3]);
            set(obj.H.timeAx, 'xlim',stepW*[-1,1]+[0,1]);
            hold(obj.H.timeAx, 'on');
            % analysis time range
            obj.H.timeRangeLine = plot(obj.H.timeAx, [0 1], 0.5*[1 1], 'c', 'LineWidth', 2*obj.H.tracelw);
            set(obj.H.timeRangeLine, 'PickableParts','none');
            % current time
            obj.H.timeLine = plot(obj.H.timeAx, [0 0], [0 1], 'g', 'LineWidth', 2*obj.H.tracelw);
            set(obj.H.timeLine, 'PickableParts','none');
            title(obj.H.timeAx, 'time in recording - click to jump');
            axis(obj.H.timeAx, 'off');
            set(obj.H.timeBckg, 'ButtonDownFcn', @(f,k)obj.timeClickCB(f,k));
            set(obj.H.timeLBtn, 'ButtonDownFcn', @(f,k)obj.timeClickLeftCB(f,k));
            set(obj.H.timeRBtn, 'ButtonDownFcn', @(f,k)obj.timeClickRightCB(f,k));            
            
            obj.H.dataVBox.Sizes = [30 -6 150];
            
            
            %% Create Message log box
            obj.H.logBox = uicontrol(...
                'Parent', obj.H.logPanel,...
                'Style', 'listbox', 'Enable', 'inactive', 'String', {}, ...
                'Tag', 'Logging Display', 'FontSize', .8*obj.H.fsz);                        
        end
        
        function initPars(obj)
            
            % get ops
            obj.ops = ksGUI.defaultOps();  
            
            obj.P.currT = 0.1;
            obj.P.tWin = [0 0.1];
            obj.P.currY = 0;
            obj.P.currX = 0;
            initChanCount = 32;
            obj.P.nChanToPlot = initChanCount;
            obj.P.nChanToPlotCM = initChanCount;
            obj.P.selChans = 1:initChanCount;            
            obj.P.vScale = 0.0001;
            obj.P.dataGood = false;
            obj.P.probeGood = false;            
            obj.P.ksDone = false;
            obj.P.preProcDone = false;
            obj.P.colormapMode = false; 
            obj.P.showRaw = true;
            obj.P.showWhitened = false;
            obj.P.showPrediction = false;
            obj.P.showResidual = false;
            obj.P.saveSettingsOnClose = false;
            
            mfPath = fileparts(mfilename('fullpath'));
            cm = load(fullfile(mfPath, 'cmap.mat')); %grey/red
            obj.P.colormap = cm.cm; 
            
            % get gui defaults/remembered settings
            obj.P.settingsPath = fullfile(mfPath, 'userSettings.mat');
            if 0 %exist(obj.P.settingsPath, 'file')
                savedSettings = load(obj.P.settingsPath);
                if isfield(savedSettings, 'lastFile')
                    obj.H.settings.ChooseFileEdt.String = savedSettings.lastFile;
                    obj.log('Initializing with last used file.');
                    try
                        obj.restoreGUIsettings();
                        obj.updateProbeView('new');
                        obj.updateFileSettings();
                    catch ex
                        obj.log('Failed to initialize last file.');
                        % keyboard
                    end
                end
            else
                obj.log('Select a data file (upper left) to begin.');
            end
            
            % obj.updateProbeView('new');
            
        end
        
        
        %% Select Data file callback
        function selectFileDlg(obj)
            [filename, pathname] = uigetfile('*.dat', 'Pick a data file.');
            
            if filename~=0 % 0 when cancel
                obj.H.settings.ChooseFileEdt.String = ...
                    fullfile(pathname, filename);
                obj.log(sprintf('Selected file %s', obj.H.settings.ChooseFileEdt.String));
                
                obj.P.ksDone = false;
                obj.P.preProcDone = false;
                
                obj.updateFileSettings();
                
                % obj.updateProbeView('new');
            end
            
        end
        
        function selectDirDlg(obj, src)
            switch src.Tag
                case 'output'
                    startDir = obj.H.settings.ChooseOutputEdt.String;
            end
            if strcmp(startDir, '...'); startDir = ''; end
            pathname = uigetdir(startDir, 'Pick a directory.');
            
            if pathname~=0 % 0 when cancel
                obj.H.settings.ChooseOutputEdt.String = pathname;
                obj.updateFileSettings();
            end
        end %selectDirDlg
        
        
        %% updateFileSettings
        function updateFileSettings(obj)
            
            % check whether there's a data file and exists
            if strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                return;
            end
            if ~exist(obj.H.settings.ChooseFileEdt.String, 'file')                     
                obj.log('Data file does not exist.');
                return;
            end
            
            
            % check file extension
            [~,~,ext] = fileparts(obj.H.settings.ChooseFileEdt.String);
            if ~strcmp(ext, '.bin') &&  ~strcmp(ext, '.dat')
                obj.log('Warning: Data file must be raw binary. Other formats not supported.');
            end
            
            % if data file exists and output/temp are empty, pre-fill
            % % %             if strcmp(obj.H.settings.ChooseTempdirEdt.String, '...')||...
            % % %                 isempty(obj.H.settings.ChooseTempdirEdt.String)
            % % %                 pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
            % % %                 obj.H.settings.ChooseTempdirEdt.String = pathname;
            % % %             end
            if strcmp(obj.H.settings.ChooseOutputEdt.String, '...')||...
                isempty(obj.H.settings.ChooseOutputEdt.String)
                pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
                obj.H.settings.ChooseOutputEdt.String = pathname;
            end
            
            nChan = obj.checkNChan();                    
                
            if ~isempty(nChan)
                % if all that looks good, make the plot
            
                obj.P.dataGood = true;
                obj.P.datMMfile = [];
                if nChan>=64
                    obj.P.colormapMode = true;
                    obj.P.nChanToPlotCM = nChan;
                end
                obj.updateDataView()

                lastFile = obj.H.settings.ChooseFileEdt.String;
                save(obj.P.settingsPath, 'lastFile');

                if obj.P.probeGood
                    set(obj.H.settings.runBtn, 'enable', 'on');
                    set(obj.H.settings.runPreprocBtn, 'enable', 'on');
                end      
            end
            obj.refocus(obj.H.settings.ChooseFileTxt);
            
        end % updateFileSettings
        
        
        %% updateParameterSettings
        function updateParameterSettings(obj)
            
            % check whether there's a data file and exists
            if strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                return;
            end
            if ~exist(obj.H.settings.ChooseFileEdt.String, 'file')                     
                obj.log('Data file does not exist.');
                return;
            end
            
            
            % check file extension
            [~,~,ext] = fileparts(obj.H.settings.ChooseFileEdt.String);
            if ~strcmp(ext, '.bin') &&  ~strcmp(ext, '.dat')
                obj.log('Warning: Data file must be raw binary. Other formats not supported.');
            end
            
            % if data file exists and output/temp are empty, pre-fill
            % % %             if strcmp(obj.H.settings.ChooseTempdirEdt.String, '...')||...
            % % %                 isempty(obj.H.settings.ChooseTempdirEdt.String)
            % % %                 pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
            % % %                 obj.H.settings.ChooseTempdirEdt.String = pathname;
            % % %             end
            if strcmp(obj.H.settings.ChooseOutputEdt.String, '...')||...
                isempty(obj.H.settings.ChooseOutputEdt.String)
                pathname = fileparts(obj.H.settings.ChooseFileEdt.String);
                obj.H.settings.ChooseOutputEdt.String = pathname;
            end
            
            nChan = obj.checkNChan();                    
                
            if ~isempty(nChan)
                % if all that looks good, make the plot
            
                obj.P.dataGood = true;
                obj.P.datMMfile = [];
                if nChan>=64
                    obj.P.colormapMode = true;
                    obj.P.nChanToPlotCM = nChan;
                end
                obj.updateDataView()

                lastFile = obj.H.settings.ChooseFileEdt.String;
                save(obj.P.settingsPath, 'lastFile');

                if obj.P.probeGood
                    set(obj.H.settings.runBtn, 'enable', 'on');
                    set(obj.H.settings.runPreprocBtn, 'enable', 'on');
                end      
            end
            obj.refocus(obj.H.settings.ChooseFileTxt);
            
        end % updateParameterSettings        
        
        
        
        %% Check & load data
        function nChan = checkNChan(obj)
            origNChan = 0;
            % if nChan is set, see whether it makes any sense
            if ~isempty(obj.H.settings.setnChanEdt.String)
                nChan = str2num(obj.H.settings.setnChanEdt.String);
                origNChan = nChan;
                if isfield(obj.P, 'chanMap') && sum(obj.P.chanMap.connected)>nChan
                    nChan = numel(obj.P.chanMap.chanMap); % need more channels
                end
                    
            elseif isfield(obj.P, 'chanMap')  
                % initial guess that nChan is the number of channels in the channel
                % map
                nChan = numel(obj.P.chanMap.chanMap);
            else
                nChan = 32; % don't have any other guess
            end
            
            if ~isempty(obj.H.settings.ChooseFileEdt.String) && ...
                    ~strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                b = get_file_size(obj.H.settings.ChooseFileEdt.String);

                a = cast(0, 'int16'); % hard-coded for now, int16
                q = whos('a');
                bytesPerSamp = q.bytes;

                if ~(mod(b,bytesPerSamp)==0 && mod(b/bytesPerSamp,nChan)==0)
                    % try figuring out number of channels, since the previous
                    % guess didn't work
                    testNC = ceil(nChan*0.9):floor(nChan*1.1);
                    possibleVals = testNC(mod(b/bytesPerSamp, testNC)==0);
                    if ~isempty(possibleVals)
                        if ~isempty(find(possibleVals>nChan,1))
                            nChan = possibleVals(find(possibleVals>nChan,1));
                        else
                            nChan = possibleVals(end);
                        end
                        obj.log(sprintf('Guessing that number of channels is %d. If it doesn''t look right, consider trying: %s', nChan, num2str(possibleVals)));
                    else
                        obj.log('Cannot work out a guess for number of channels in the file. Please enter number of channels to proceed.');
                        nChan = [];
                        return;
                    end
                end
                obj.H.settings.setnChanEdt.String = num2str(nChan);
                obj.P.nSamp = b/bytesPerSamp/nChan;
                
                
            end
            
            if nChan~=origNChan
                obj.P.datMMfile = [];
                if nChan>=64
                    obj.P.colormapMode = true;
                    obj.P.nChanToPlotCM = nChan;
                end
            end
        end
        
        
        %% addNewChanMap
        function addNewChanMap(obj, cm)
            % add new chanMap struct to existing amalgamation of all chanMaps in Kilosort Config dir.
            % - bandaid/workaround for convention of holding ALL chan maps in current gui struct at all times
            % - previously, constrained chan maps to limited set of fields/info
            fnNew = fieldnames(cm);
            fnAll = fieldnames(obj.P.allChanMaps);
            if all(ismember(fnNew,fnAll))
                obj.P.allChanMaps(end+1) = cm;
            else
                ii = length(obj.P.allChanMaps)+1;
                for i = 1:length(fnNew)
                    obj.P.allChanMaps(ii).(fnNew{i}) = cm.(fnNew{i});
                end
            end
            
        end %addNewChanMap
        
        
        %% Advanced Options popup
        function advancedPopup(obj)
            % move focus to command window
            commandwindow;
            % bring up popup window to set other ops
            helpdlg({'To set advanced options, do this in the command window:','',...
                '>> ks = get(gcf, ''UserData'');',...
                sprintf('\t[...I''ll do this for you now]'),...
                '>> ks.ops.myOption = myValue;'});
            evalin('base', sprintf('ks = get(%d, ''UserData'');',obj.H.fig.Number)) % 1029321 is default kilosort gui window figure number
            evalin('base', 'fprintf(2,''ks.ops =\n''), disp(ks.ops)'); % show current parameters in command window

        end %advancedPopup
        
        
        %% prepareForRun
        function prepareForRun(obj)
            % CATCHALL check that everything is set up correctly to run
            
            obj.ops.fbinary = obj.H.settings.ChooseFileEdt.String;
            if ~exist(obj.ops.fbinary, 'file')
                obj.log('Cannot run: Data file not found.');
                return;
            end
                        
            obj.ops.saveDir = obj.H.settings.ChooseOutputEdt.String;
            if ~exist(obj.ops.saveDir, 'dir') && ~contains(obj.ops.saveDir,'...')
                mkdir(obj.ops.saveDir);
            end
            
            % working directory & save directory are one and the same
            [~,obj.ops.fname] = fileparts(obj.ops.saveDir); 
            obj.ops.fproc = fullfile(obj.ops.saveDir, sprintf('proDat_%s.dat',obj.ops.fname)); %'temp_wh.dat');
            
            % build channel map that includes only the connected channels
            % % Stop overwriting the chanMap struct!!   chanMap = struct();
            chanMap = obj.P.chanMap;
            conn = obj.P.chanMap.connected;
            chanMap.chanMap = obj.P.chanMap.chanMap(conn); 
            chanMap.xcoords = obj.P.chanMap.xcoords(conn); 
            chanMap.ycoords = obj.P.chanMap.ycoords(conn);
            if isfield(obj.P.chanMap, 'kcoords')
                chanMap.kcoords = obj.P.chanMap.kcoords(conn);
            end
            % Stop overwriting the chanMap struct!!
            obj.ops.chanMap = chanMap;
            
            % sanitize options set in the gui
            % - .Nfilt must be updated when user right-clicks any channel to 'disconnnect it',
            %   making it tricky to cleanly parameterize
            nFF = getOr(obj.ops, 'nfilt_factor',4);
            obj.ops.Nfilt = numel(obj.ops.chanMap.chanMap) * nFF;
            
            if ~isfield(obj.ops,'Nfilt') || isempty(obj.ops.Nfilt) || isnan(obj.ops.Nfilt)
                obj.ops.Nfilt = numel(obj.ops.chanMap.chanMap)*nFF;
            elseif obj.ops.Nfilt > numel(obj.ops.chanMap.chanMap)*nFF
                obj.log('~!~ Warning:  max templates parameter [ops.Nfilt] exceeds 4*nChannels...this could be problematic/slow');
            end
            
            if mod(obj.ops.Nfilt,32)~=0
                obj.ops.Nfilt = ceil(obj.ops.Nfilt/32)*32;
            end
            
            obj.ops.NchanTOT = str2double(obj.H.settings.setnChanEdt.String);
            
            obj.ops.nblocks = str2double(obj.H.settings.setMinfrEdt.String);
            if isempty(obj.ops.nblocks)||isnan(obj.ops.nblocks)
                obj.ops.nblocks = 1;
            end
            obj.ops.throw_out_channels = false;
            obj.H.settings.setMinfrEdt.String = num2str(obj.ops.nblocks);

            obj.ops.fs = str2num(obj.H.settings.setFsEdt.String);
            if isempty(obj.ops.fs)||isnan(obj.ops.fs)
                obj.ops.fs = 40000;
            end
                        
            obj.ops.Th = str2num(obj.H.settings.setThEdt.String);
            if isempty(obj.ops.Th)||any(isnan(obj.ops.Th))
                obj.ops.Th = [10 4];
            end
            obj.H.settings.setThEdt.String = num2str(obj.ops.Th);
            
            obj.ops.lam = str2num(obj.H.settings.setLambdaEdt.String);
            if isempty(obj.ops.lam)||isnan(obj.ops.lam)
                obj.ops.lam = 10;
            end
            obj.H.settings.setLambdaEdt.String = num2str(obj.ops.lam);
            
            obj.ops.AUCsplit = str2double(obj.H.settings.setCcsplitEdt.String);
            if isempty(obj.ops.AUCsplit)||isnan(obj.ops.AUCsplit)
                obj.ops.AUCsplit = 0.9;
            end
            obj.H.settings.setCcsplitEdt.String = num2str(obj.ops.AUCsplit);
            
            obj.ops.trange = str2num(obj.H.settings.setTrangeEdt.String);
            if isempty(obj.ops.trange)||any(isnan(obj.ops.trange))
                obj.ops.trange = [0 Inf];
            end
            obj.H.settings.setTrangeEdt.String = num2str(obj.ops.trange);
            
        end %prepareForRun
        
        
        %% [Run All] button callback
        function runAll(obj)
            obj.P.ksDone = false;
            obj.P.preProcDone = false;
            
            obj.runPreproc;
            obj.runSpikesort;
            obj.runSaveToPhy;
            
        end %runAll
        
        
        %% [Sort & Save] button callback
        function runSortAndSave(obj)
            if ~obj.P.preProcDone
                fprintf(2, 'PreProcessing must be completed before Run & Save\n\tTry again after doing [Preprocessing] or use [Run All]')
                return
            else
                obj.P.ksDone = false;
                
                obj.runSpikesort;
                obj.runSaveToPhy;
            end
        end %runSortAndSave
        
                
        %% [Preprocess] button callback
        function runPreproc(obj)
            obj.prepareForRun;
            
            % do preprocessing
            obj.ops.gui = obj; % for kilosort to access, e.g. calling "log"
%             try
                obj.log('Preprocessing...'); 
                obj.rez = preprocessDataSub(obj.ops);  % ks25 version updated to write file from t0 regardless of tstart; get_batch will do the rest
                
                % apply temporal alignment
                obj.rez = datashift2(obj.rez, 1);
                
                % update connected channels
                igood = obj.rez.ops.igood;
                previousGood = find(obj.P.chanMap.connected);
                newGood = previousGood(igood);
                obj.P.chanMap.connected = false(size(obj.P.chanMap.connected));
                obj.P.chanMap.connected(newGood) = true;
                
                % use the new whitening matrix, which can sometimes be
                % quite different than the earlier estimated one
                rW = obj.rez.Wrot;
                if isfield(obj.P, 'Wrot')                    
                    pW = obj.P.Wrot;                    
                else
                    pW = zeros(numel(obj.P.chanMap.connected));
                end
                cn = obj.P.chanMap.connected;
                pW(cn,cn) = rW;
                obj.P.Wrot = pW;
                
                set(obj.H.settings.runSpikesortBtn, 'enable', 'on');
                set(obj.H.settings.runSortSaveBtn, 'enable', 'on');
                
                % Update spikesorting run status flag
                obj.P.ksDone = false;
                obj.P.preProcDone = true;
                % clear GUI mmf object; will automatically be recreated to ensure consistent w/ ops.trange
                obj.P.datMMfile = []; 
                obj.P.showWhitened = true;
                
                % update gui with results of preprocessing
                obj.updateDataView();
                obj.log('Done preprocessing.'); 
%             catch ex
%                 obj.log(sprintf('Error preprocessing! %s', ex.message));
%                 keyboard
%             end
            
        end %runPreproc
        
        
        %% computeWhitening
        function computeWhitening(obj)
            obj.log('Computing whitening filter...')
            obj.prepareForRun;
            
            % here, use a different channel map than the actual: show all
            % channels as connected. That way we can drop/add without
            % recomputing everything later. 
            ops = obj.ops;
            ops.chanMap = obj.P.chanMap;
            ops.chanMap.connected = true(size(ops.chanMap.connected));
            
            [~,Wrot] = computeWhitening(ops); % this refers to a function outside the gui
            
            obj.P.Wrot = Wrot;
            obj.updateDataView;
            obj.log('Done.')
            
        end %computeWhitening
        
        
        %% [Spikesort] button callback
        function runSpikesort(obj)
            % fit templates
%             try
                
                % main optimization
                obj.log('Main optimization')
                % Ensure a new learn & extraction is performed when spike sort button is clicked
                % - ...never a circumstance where using GUI & initializing with a pre-existing set of waveform templates
                obj.rez.W = [];
                obj.rez.U = [];
                obj.rez.mu = [];
                obj.rez = learnAndSolve8b(obj.rez, now);
                
                % final splits and merges
                if 1
                    obj.log('--- Beginning post-processing...')
                    
                    % OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
                    % See issue 29: https://github.com/MouseLand/Kilosort/issues/29
                    if getOr(obj.rez.ops, 'rmDuplicates', 0)
                        obj.log('Pruning duplicate spikes...')
                        obj.rez = remove_ks2_duplicate_spikes(obj.rez);
                    else
                        obj.rez.ops.rmDuplicates = 0;
                    end
                    
                    % Find Merges
                    obj.log('Merges...')
                    % flag==2 ignores potential refractory violations when deciding on merges
                    % - when templates pass similarity crit, these are often double counted spikes, which can (& should) be readily detected & removed later
                    obj.rez = find_merges(obj.rez, 1);
                    
                    % Find Splits
                    if any(obj.rez.ops.splitClustersBy==1)
                        % final splits by SVD
                        obj.log('Splitting clusters (by template projections)...')
                        obj.rez = splitAllClusters(obj.rez, 1);
                    end
                    if any(obj.rez.ops.splitClustersBy==2)
                        % final splits by waveform amplitude
                        obj.log('Splitting clusters (by amplitudes)...')
                        obj.rez = splitAllClusters(obj.rez, 0);
                    end
                    

                    % Determine (& apply) cutoff
                    if getOr(obj.rez.ops, 'applyCutoff', 1)
                        obj.rez.ops.applyCutoff = 1;
                        obj.log('Setting threshold cutoff (with removal) & estimating contamination...')
                    else
                        obj.rez.ops.applyCutoff = 0;
                        obj.log('Setting threshold cutoff (WITHOUT removal) & estimating contamination...')
                    end
                    obj.rez = set_cutoff(obj.rez, obj.rez.ops.applyCutoff);
                    
                    [obj.rez.good, status] = get_good_units(obj.rez);
                    
                    obj.log(status)
                end
                                                                
                obj.P.ksDone = true;
                
                % reset gui data source file so new fproc data file will be memmapped
                if isfield(obj.P, 'datMMfile') && ~isempty(obj.P.datMMfile)
                    obj.P.datMMfile = [];
                    % clear gui whitening matrix since post-sort data source is already whitened (temp_wh.dat)
                    obj.P.Wrot = [];
                end
                
                obj.log('Kilosort finished!');
                set(obj.H.settings.runSaveBtn, 'enable', 'on');
                obj.updateDataView();
                % bring kilosort figure to focus
                figure(obj.H.fig)
%             catch ex
%                 obj.log(sprintf('Error running kilosort! %s', ex.message));
%             end   
                        
        end %runSpikesort
        
        
        %% [Save for Phy] button callback
        function runSaveToPhy(obj)
            tic
            % save results
            obj.log(sprintf('Saving data to %s', obj.ops.saveDir));
            % thin out rez struct before saving
            rez = obj.rez;
            rez.ops.gui = [];
            
            % Remove certain [large] fields before saving rez struct to mat
            fprintf('\n---\tPruning fields from rez struct before saving as .mat:')
            theseFields = {};
            % .cProj & .cProjPC     remove feature projections from rez (default of original ksGUI.m)
            % - these are REALLY BIG (nspikes x 32; easily 100s of MB to multiple GB on their own)
            % - these get written out as template_features.npy & pc_features.npy during rezToPhy, so they are not lost by discarding them here
            theseFields = [theseFields, 'cProj', 'cProjPC'];
%             % .WA & .UA     temporally dynamic template shape and channel weighting matrices    (.WA == [nt0, nUnits, nPC, nBatches]; .UA == [nChan, nUnits, nPC, nBatches])
%             theseFields = [theseFields, 'WA', 'UA'];  
            
            for i = 1:length(theseFields)
                ii = theseFields{i};
                if isfield(rez, ii) && ~isempty(rez.(ii))
                    fprintf('\n\t%s', ii),
                    rez.(ii) = [];
                else
                    fprintf('\n\t%s\t(**not found, or already empty)', ii),
                end
            end
            fprintf('\n---\tDone.\n');

            % Ensure all GPU arrays are transferred to CPU side before saving to .mat
            rez_fields = fieldnames(rez);
            for i = 1:numel(rez_fields)
                field_name = rez_fields{i};
                if(isa(rez.(field_name), 'gpuArray'))
                    rez.(field_name) = gather(rez.(field_name));
                end
            end

            % save final results as rez.mat
            fname = fullfile(obj.ops.saveDir, 'rez.mat');
            save(fname, 'rez', '-v7.3');

            % save .mat copy of chanmap
            chanMap = obj.ops.chanMap;
            save(fullfile(obj.ops.saveDir,'chanMap.mat'), 'chanMap');
            
            % save ops struct (...avoid loading ENTIRE rez struct just to check a sort parameter)
            oo = rez.ops;
            % sort fields alphabetically [human-readable]
            fn = fieldnames(oo);
            [~, fni] = sort(lower(fn));
            oo = orderfields(oo, fni);
            save(fullfile(obj.ops.saveDir,'opsStruct.mat'), '-struct','oo');
            
            % Create & save Phy outputs
            rezToPhy(obj.rez, obj.ops.saveDir);
            toc

            obj.log('Done');
            
        end %runSaveToPhy
        
        
        %% updateDataView
        function updateDataView(obj)
            
            if obj.P.dataGood && obj.P.probeGood

                % get currently selected time and channels                
                t = obj.P.currT;
                % error check selected channels with current chanMap
                if any(obj.P.selChans>numel(obj.P.chanMap.chanMap))
                    obj.P.selChans = 1:max(numel(obj.P.chanMap.chanMap), 32);   %16);
                end
                chList = obj.P.selChans;
                tWin = obj.P.tWin;
                
                % initialize data loading if necessary
                if ~isfield(obj.P, 'datMMfile') || isempty(obj.P.datMMfile)
                    if obj.P.ksDone
                        % use preprocessed whitened data so that drift correction is accounted for
                        filename = obj.ops.fproc;
                    else
                        filename = obj.H.settings.ChooseFileEdt.String;
                    end
                    datatype = 'int16';
                    chInFile = str2double(obj.H.settings.setnChanEdt.String);
                    b = get_file_size(filename);
                    obj.P.nSamp = b/chInFile/2; % 'int16' datatype, might as well hard-code corresponding bytesPerSamp
                    nSamp = obj.P.nSamp;
                    mmf = memmapfile(filename, 'Format', {datatype, [chInFile nSamp], 'x'});
                    obj.P.datMMfile = mmf;
                    obj.P.datSize = [chInFile nSamp];
                else
                    mmf = obj.P.datMMfile;
                end
                
                Fs = str2double(obj.H.settings.setFsEdt.String);
                samps = ceil(Fs*(t+tWin));
                if all(samps>0 & samps<obj.P.datSize(2))
                    
                    % load and process data
                    datAll = mmf.Data.x(:,samps(1):samps(2));
                    
                    % filtered, whitened
                    obj.prepareForRun();
                    datAllF = ksFilter(datAll, obj.ops);
                    datAllF = double(gather(datAllF));
                    if isfield(obj.P, 'Wrot') && ~isempty(obj.P.Wrot)
                        %Wrot = obj.P.Wrot/obj.ops.scaleproc;
                        conn = obj.P.chanMap.connected;
                        Wrot = obj.P.Wrot(conn,conn);
                        if obj.P.ksDone
                             Wrot = Wrot / obj.ops.scaleproc;
                             %                         else
                             %                             Wrot = Wrot / sqrt(obj.ops.scaleproc);
                        end 
                        datAllF = datAllF * Wrot;
                    end
                    datAllF = datAllF';
                    
                    if obj.P.ksDone && isfield(obj.rez, 'W')
                        % scale up if using preprocessed data
                        pd = predictData(obj.rez, samps);
                    else
                        pd = zeros(size(datAllF));
                    end
                    
                    dat = datAll(obj.P.chanMap.chanMap(chList),:);
                    
                    connInChList = obj.P.chanMap.connected(chList);
                    
                    cmconn = obj.P.chanMap.chanMap(obj.P.chanMap.connected);
                                        
                    chListW = NaN(size(chList)); % channels within the processed data
                    for q = 1:numel(chList)
                        if obj.P.chanMap.connected(chList(q))
                            chListW(q) = find(cmconn==chList(q),1);
                        end
                    end
                    
                    datW = zeros(numel(chList),size(datAll,2));
                    datP = zeros(numel(chList),size(datAll,2));
                    datW(~isnan(chListW),:) = datAllF(chListW(~isnan(chListW)),:);
                    datP(~isnan(chListW),:) = pd(chListW(~isnan(chListW)),:);
                    datR = datW-datP;
                    
                    
                    % ------ Plot Data -------
                    if ~obj.P.colormapMode
                        %% Data Axis: Colormap Mode
                        % data axis title string
                        ttl = '';
                        
                        if isfield(obj.H, 'dataIm') && ~isempty(obj.H.dataIm)
                            set(obj.H.dataIm, 'Visible', 'off');
                        end
                        
                        % Raw data traces
                        if obj.P.showRaw    % obj.H.dataTr
                            % data axis title string                            
                            ttl = [ttl '\color[rgb]{0 0 0}raw '];
                            
                            if ~isfield(obj.H, 'dataTr') || numel(obj.H.dataTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'dataTr')&&~isempty(obj.H.dataTr);
                                    delete(obj.H.dataTr);
                                end
                                obj.H.dataTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.dataTr(q) = plot(obj.H.dataAx, 0, NaN, 'k', 'LineWidth', obj.H.tracelw);
                                    set(obj.H.dataTr(q), 'HitTest', 'off');
                                end
                                box(obj.H.dataAx, 'off');
                                
                            end                                                

                            conn = obj.P.chanMap.connected(chList);
                            for q = 1:size(dat,1)
                                set(obj.H.dataTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                    'YData', q+double(dat(q,:)).*obj.P.vScale,...
                                    'Visible', 'on');
                                if conn(q); set(obj.H.dataTr(q), 'Color', 'k');
                                else; set(obj.H.dataTr(q), 'Color', 0.8*[1 1 1]); end
                            end                                                                        
                        elseif isfield(obj.H, 'dataTr')
                            for q = 1:numel(obj.H.dataTr)
                                set(obj.H.dataTr(q), 'Visible', 'off');
                            end
                        end
                        
                        
                        % Filtered data traces
                        if obj.P.showWhitened   % obj.H.ppTr
                            % data axis title string
                            ttl = [ttl '\color[rgb]{0 0.6 0}filtered '];
                            
                            if ~isfield(obj.H, 'ppTr') || numel(obj.H.ppTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'ppTr')&&~isempty(obj.H.ppTr); delete(obj.H.ppTr); end
                                obj.H.ppTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    % brighten filtered data trace for clear luminance difference btwn filtered & prediction
                                    obj.H.ppTr(q) = plot(obj.H.dataAx, 0, NaN, 'Color', [0 0.6 0]+0.2, 'LineWidth', obj.H.tracelw);
                                    set(obj.H.ppTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.ppTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datW(q,:).*obj.P.vScale);
                                end
                            end
                        elseif isfield(obj.H, 'ppTr')
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'off');
                            end
                        end
                        
                        
                        % Prediction data traces
                        if obj.P.showPrediction   % obj.H.predTr
                            % data axis title string
                            ttl = [ttl '\color[rgb]{0 0 1}prediction '];
                            
                            if ~isfield(obj.H, 'predTr') || numel(obj.H.predTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'predTr') && ~isempty(obj.H.predTr)
                                    delete(obj.H.predTr);
                                end
                                obj.H.predTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.predTr(q) = plot(obj.H.dataAx, 0, NaN, 'b', 'LineWidth', obj.H.tracelw);
                                    set(obj.H.predTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.predTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datP(q,:).*obj.P.vScale);
                                end
                            end
                        elseif isfield(obj.H, 'predTr')
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'off');
                            end
                        end
                        
                        
                        % Residual data traces   
                        if obj.P.showResidual   % obj.H.residTr
                            % data axis title string
                            ttl = [ttl '\color[rgb]{1 0 0}residual '];
                            
                            if ~isfield(obj.H, 'residTr') || numel(obj.H.residTr)~=numel(chList)
                                % initialize traces
                                if isfield(obj.H, 'residTr')&&~isempty(obj.H.residTr); delete(obj.H.residTr); end
                                obj.H.residTr = [];
                                hold(obj.H.dataAx, 'on');
                                for q = 1:numel(chList)
                                    obj.H.residTr(q) = plot(obj.H.dataAx, 0, NaN, 'r', 'LineWidth', obj.H.tracelw);
                                    set(obj.H.residTr(q), 'HitTest', 'off');
                                end
                            end
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'on');
                            end

                            for q = 1:numel(chListW)  
                                if ~isnan(chListW(q))
                                    set(obj.H.residTr(q), 'XData', (samps(1):samps(2))/Fs, ...
                                        'YData', q+datR(q,:).*obj.P.vScale);
                                end
                            end
                        elseif isfield(obj.H, 'residTr')
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'off');
                            end
                        end
                        
                        % data axis title string
                        set(obj.H.dataAx.Title, 'String',ttl);
                        % arrange data trace layering  [raw, filtered, predicted, residual]
                        daxTr = [obj.H.residTr,  obj.H.predTr, obj.H.ppTr, obj.H.dataTr];
                        daxCh = get(obj.H.dataAx, 'Children');
                        xtraH = daxCh(~ismember(daxCh, daxTr));  % allow any additional/future overlays
                        set(obj.H.dataAx, 'Children', [xtraH(:); daxTr(:)]);

                        yt = arrayfun(@(x)sprintf('%d (%d)', chList(x), obj.P.chanMap.chanMap(chList(x))), 1:numel(chList), 'uni', false);
                        set(obj.H.dataAx, 'YTick', 1:numel(chList), 'YTickLabel', yt);
                        set(obj.H.dataAx, 'YLim', [0 numel(chList)+1], 'YDir', 'normal');
                        
                    else
                        %% Data Axis: Colormap Mode
                        %chList = 1:numel(obj.P.chanMap.chanMap);
                        
                        if ~isfield(obj.H, 'dataIm') || isempty(obj.H.dataIm)
                            obj.H.dataIm = [];
                            hold(obj.H.dataAx, 'on');
                            obj.H.dataIm = imagesc(obj.H.dataAx, chList, ...
                                (samps(1):samps(2))/Fs,...
                                datAll(obj.P.chanMap.connected,:));
                            set(obj.H.dataIm, 'HitTest', 'off');                            
                            colormap(obj.H.dataAx, obj.P.colormap);
                        end
                        
                        if isfield(obj.H, 'dataTr') && ~isempty(obj.H.dataTr)
                            for q = 1:numel(obj.H.dataTr)
                                set(obj.H.dataTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'ppTr') && ~isempty(obj.H.ppTr)
                            for q = 1:numel(obj.H.ppTr)
                                set(obj.H.ppTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'predTr') && ~isempty(obj.H.predTr)
                            for q = 1:numel(obj.H.predTr)
                                set(obj.H.predTr(q), 'Visible', 'off');
                            end
                        end
                        if isfield(obj.H, 'residTr') && ~isempty(obj.H.residTr)
                            for q = 1:numel(obj.H.residTr)
                                set(obj.H.residTr(q), 'Visible', 'off');
                            end
                        end
                        
                        set(obj.H.dataIm, 'Visible', 'on');
                        if obj.P.showRaw
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', dat(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*1000);   %*obj.P.vScale*15000);
                            title(obj.H.dataAx, 'raw');
                        elseif obj.P.showWhitened
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datW(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*1000);   %*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'filtered');
                        elseif obj.P.showPrediction
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datP(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*1000);   %*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'prediction');
                        else % obj.P.showResidual
                            set(obj.H.dataIm, 'XData', (samps(1):samps(2))/Fs, ...
                                'YData', 1:sum(connInChList), 'CData', datR(connInChList,:));                    
                            set(obj.H.dataAx, 'CLim', [-1 1]*obj.P.vScale*1000);   %*obj.P.vScale*225000);
                            title(obj.H.dataAx, 'residual');
                        end
                        set(obj.H.dataAx, 'YLim', [0 sum(connInChList)]+0.5, 'YDir', 'normal');
                    end
                    
                    set(obj.H.dataAx, 'XLim', t+tWin);
                    
                    set(obj.H.dataAx, 'YTickLabel', []);
                end
                
                % update time axis (trange)
                updateDataTimeAx(obj);
                
            end
        end % updateDataView
        
        
        %% updateDataTimeAx
        function updateDataTimeAx(obj, varargin)
            % update time axis
                nSamp = obj.P.nSamp;
                maxT = nSamp/obj.ops.fs;
                atr = str2num(obj.H.settings.setTrangeEdt.String)/maxT;
                atr(atr<0) = 0; atr(atr>1) = 1;
                if isempty(atr)
                    atr = [0,1];
                end
                set(obj.H.timeRangeLine, 'XData', atr);
        end %updateDataTimeAx
        
        
        %% updateProbeView
        function updateProbeView(obj, varargin)
            
            if ~isempty(varargin) % any argument means to re-initialize
            
                obj.P.probeGood = false; % might have just selected a new one

                % if probe file exists, load it
                selProbe = obj.H.settings.setProbeEdt.String{obj.H.settings.setProbeEdt.Value};

                cm = [];
                switch selProbe
                    case '[new]'
                        %obj.log('New probe creator not yet implemented.');
                        answer = inputdlg({'Name for new channel map:', ...
                            'X-coordinates of each site (can use matlab expressions):',...
                            'Y-coordinates of each site:',...
                            'Shank index (''kcoords'') for each site (blank for single shank):',...
                            'Channel map (the list of rows in the data file for each site):',...
                            'List of disconnected/bad site numbers (blank for none):'});
                        if isempty(answer)
                            return;
                        else
                            cm.name = answer{1};
                            cm.xcoords = str2num(answer{2});
                            cm.ycoords = str2num(answer{3});
                            if ~isempty(answer{4})
                                cm.kcoords = str2num(answer{4});
                            end
                            cm.chanMap = str2num(answer{5});
                            if ~isempty(answer{6})
                                q = str2num(answer{6});
                                if numel(q) == numel(cm.chanMap)
                                    cm.connected = q;
                                else
                                    cm.connected = true(size(cm.chanMap));
                                    cm.connected(q) = false;
                                end
                            end
                            cm = createValidChanMap(cm);
                            if ~isempty(cm)
                                answer = questdlg('Save this channel map for later?');
                                if strcmp(answer, 'Yes')
                                    cm = saveNewChanMap(cm, obj);
                                end
                                addNewChanMap(obj, cm);
                                % obj.P.allChanMaps(end+1) = cm;
                                currProbeList = obj.H.settings.setProbeEdt.String;
                                newProbeList = [{cm.name}; currProbeList];
                                obj.H.settings.setProbeEdt.String = newProbeList;
                                obj.H.settings.setProbeEdt.Value = 1;
                                % import settings from chanMap to gui
                                if isfield(cm,'fs') && ~isempty(cm.fs)
                                    obj.H.settings.setFsEdt.String = num2str(cm.fs);  % continuous sampling rate
                                end
                            else
                                obj.log('Channel map invalid. Must have chanMap, xcoords, and ycoords of same length');
                                return;
                            end
                        end
                    case 'other...'
                        [filename, pathname] = uigetfile('*.mat', 'Pick a channel map file.');
                        
                        if filename~=0 % 0 when cancel
                            cm = load(fullfile(pathname, filename));
                            cm = createValidChanMap(cm, filename);
                            if ~isempty(cm)
                                answer = questdlg('Save this channel map for later?');
                                if strcmp(answer, 'Yes')
                                    cm = saveNewChanMap(cm, obj);
                                end
                                addNewChanMap(obj,cm);
                                currProbeList = obj.H.settings.setProbeEdt.String;
                                newProbeList = [{cm.name}; currProbeList];
                                obj.H.settings.setProbeEdt.String = newProbeList;
                                obj.H.settings.setProbeEdt.Value = 1;
                                % import settings from chanMap to gui
                                if isfield(cm,'fs') && ~isempty(cm.fs)
                                    obj.H.settings.setFsEdt.String = num2str(cm.fs);  % continuous sampling rate
                                end
                            else
                                obj.log('Channel map invalid. Must have chanMap, xcoords, and ycoords of same length');
                                return;
                            end
                        else
                            return;
                        end
                    otherwise
                        probeNames = {obj.P.allChanMaps.name};
                        cm = obj.P.allChanMaps(strcmp(probeNames, selProbe));
                end               
                % remove any empty fields from chanMap struct
                fn = fieldnames(cm);
                emptyField = cellfun(@(x) isempty(cm.(x)), fn);
                if any(emptyField)
                    cm = rmfield(cm, fn{emptyField});
                end
                
                nSites = numel(cm.chanMap);
                ux = unique(cm.xcoords); uy = unique(cm.ycoords);
                
                if isfield(cm, 'siteSize') && ~isempty(cm.siteSize)
                    ss = cm.siteSize(1);
                else
                    ss = min([diff(uy); diff(ux)]);
                end
                cm.siteSize = ss;
                               
                if isfield(obj.H, 'probeSites')&&~isempty(obj.H.probeSites)
                    delete(obj.H.probeSites);
                    obj.H.probeSites = [];
                end
                
                obj.P.chanMap = cm;
                obj.P.probeGood = true;
                
                nChan = checkNChan(obj);
                
                if obj.P.dataGood
                    obj.computeWhitening()

                    set(obj.H.settings.runBtn, 'enable', 'on');
                    set(obj.H.settings.runPreprocBtn, 'enable', 'on');
                end
            end
            
            if obj.P.probeGood
                cm = obj.P.chanMap;
                
                if ~isempty(cm)
                    % if it is valid, plot it
                    nSites = numel(cm.chanMap);
                    ux = unique(cm.xcoords); uy = unique(cm.ycoords);
                    % appropriately size site markers in probe view
                    ss = 0.8*max( [cm.siteSize, min([diff(uy); diff(ux)])] );
                    
                    if ~isfield(obj.H, 'probeSites') || isempty(obj.H.probeSites) || ...
                            numel(obj.H.probeSites)~=nSites 
                        obj.H.probeSites = [];
                        hold(obj.H.probeAx, 'on');                    
                        sq = ss*([0 0; 0 1; 1 1; 1 0]-[0.5 0.5]);
                        for q = 1:nSites
                            % plot site
                            obj.H.probeSites(q) = fill(obj.H.probeAx, ...
                                sq(:,1)+cm.xcoords(q), ...
                                sq(:,2)+cm.ycoords(q), 'b');
                            % label channel #
                            obj.H.probeSitesLabl(q) = text(obj.H.probeAx, ...
                                cm.xcoords(q), cm.ycoords(q), sprintf('%2d',q));
                            set(obj.H.probeSitesLabl(q), 'FontName','mono' ,'fontsize',.6*obj.H.fsz, ...
                                'HorizontalAlignment','center', 'clipping','on')
                            set([obj.H.probeSites(q), obj.H.probeSitesLabl(q)], 'HitTest', 'off');
                        end
                        yc = cm.ycoords;
                        ylim(obj.H.probeAx, mean(yc)+range(yc)*.52*[-1,1]); %[min(yc) max(yc)]);
                        axis(obj.H.probeAx, 'equal');
                        set(obj.H.probeAx, 'XTick', [], 'YTick', []);
                        title(obj.H.probeAx, {'scroll to zoom, click to view channel,', 'right-click to disable channel'});
                        axis(obj.H.probeAx, 'on');
                    end

                    y = obj.P.currY;
                    x = obj.P.currX;
                    if obj.P.colormapMode
                        nCh = obj.P.nChanToPlotCM;
                    else
                        nCh = obj.P.nChanToPlot;
                    end
                    if nCh>numel(cm.chanMap)
                        nCh = numel(cm.chanMap);
                    end
                    conn = obj.P.chanMap.connected;

                    dists = ((cm.xcoords-x).^2 + (cm.ycoords-y).^2).^(0.5);
                    [~, ii] = sort(dists);
                    obj.P.selChans = ii(1:nCh);

                    % order by y-coord
                    yc = obj.P.chanMap.ycoords;
                    xc = obj.P.chanMap.xcoords;
                    theseYC = [yc(obj.P.selChans), xc(obj.P.selChans)];
                    [~,ii] = sortrows(theseYC,[1,-2]); % sort by Y, use X as secondary ordering
                    obj.P.selChans = obj.P.selChans(ii);
                    
                    for q = 1:nSites
                        if ismember(q, obj.P.selChans) && ~conn(q)
                            set(obj.H.probeSites(q), 'FaceColor', [1 1 0]);
                        elseif ismember(q, obj.P.selChans) 
                            set(obj.H.probeSites(q), 'FaceColor', [0 1 0]);
                        elseif ~conn(q)
                            set(obj.H.probeSites(q), 'FaceColor', [1 0 0]);
                        else
                            set(obj.H.probeSites(q), 'FaceColor', [0 0 1]);
                        end
                    end
                    
                end
            end
        end
        
        function scrollCB(obj,~,evt)
            
            if obj.isInLims(obj.H.dataAx)
                % in data axis                
                if obj.P.dataGood
                    m = get(obj.H.fig,'CurrentModifier');

                    if isempty(m)  
                        % scroll in time
                        maxT = obj.P.nSamp/obj.ops.fs;
                        winSize = diff(obj.P.tWin);
                        shiftSize = -evt.VerticalScrollCount*winSize*0.1;
                        obj.P.currT = obj.P.currT+shiftSize;
                        if obj.P.currT>maxT; obj.P.currT = maxT; end
                        if obj.P.currT<0; obj.P.currT = 0; end
                         
                    elseif strcmp(m, 'shift')
                        % zoom in time
                        maxT = obj.P.nSamp/obj.ops.fs;
                        oldWin = obj.P.tWin+obj.P.currT;
                        newWin = ksGUI.chooseNewRange(oldWin, ...
                            1.2^evt.VerticalScrollCount,...
                            diff(oldWin)/2+oldWin(1), [0 maxT]);
                        obj.P.tWin = newWin-newWin(1);
                        obj.P.currT = newWin(1);
                        
                    elseif strcmp(m, 'control')
                        if obj.P.colormapMode
                            % zoom in Y when in colormap mode
                            obj.P.nChanToPlotCM = round(obj.P.nChanToPlotCM*1.2^evt.VerticalScrollCount);
                            if obj.P.nChanToPlotCM>numel(obj.P.chanMap.chanMap)
                                obj.P.nChanToPlotCM=numel(obj.P.chanMap.chanMap);
                            elseif obj.P.nChanToPlotCM<1
                                obj.P.nChanToPlotCM=1;
                            end
                        else
                            % scroll in channels when in traceview
                            obj.P.currY = obj.P.currY-evt.VerticalScrollCount*...
                                min(diff(unique(obj.P.chanMap.ycoords)));
                            yc = obj.P.chanMap.ycoords;
                            mx = max(yc)+obj.P.chanMap.siteSize;
                            mn = min(yc)-obj.P.chanMap.siteSize;
                            if obj.P.currY>mx; obj.P.currY = mx; end
                            if obj.P.currY<mn; obj.P.currY = mn; end
                        end
                        obj.updateProbeView();
                        
                        
                    elseif strcmp(m, 'alt')
                        % zoom in scaling of traces
                        obj.P.vScale = obj.P.vScale*1.2^(-evt.VerticalScrollCount);
                        
                    end
                    obj.updateDataView();
                end
            elseif obj.isInLims(obj.H.probeAx)
                % in probe axis
                if obj.P.probeGood
                    % Apply probe zoom to appropriate axis
                    if range(obj.P.chanMap.ycoords)>range(obj.P.chanMap.xcoords)
                        % ylim dominant
                        cpP = get(obj.H.probeAx, 'CurrentPoint');
                        yl = get(obj.H.probeAx, 'YLim');
                        currY = cpP(1,2);
                        yc = obj.P.chanMap.ycoords;
                        mx = max(yc)+obj.P.chanMap.siteSize;
                        mn = min(yc)-obj.P.chanMap.siteSize;
                        newyl = ksGUI.chooseNewRange(yl, ...
                            1.2^evt.VerticalScrollCount,...
                            currY, [mn mx]);
                        set(obj.H.probeAx, 'YLim', newyl);
                    else
                        % xlim dominant
                        cpP = get(obj.H.probeAx, 'CurrentPoint');
                        xl = get(obj.H.probeAx, 'XLim');
                        currX = cpP(1,1);%?? cpP(1,2);
                        xc = obj.P.chanMap.xcoords;
                        mx = max(xc)+obj.P.chanMap.siteSize;
                        mn = min(xc)-obj.P.chanMap.siteSize;
                        newxl = ksGUI.chooseNewRange(xl, ...
                            1.2^evt.VerticalScrollCount,...
                            currX, [mn mx]);
                        set(obj.H.probeAx, 'XLim', newxl);
                    end
                end
            end
        end
        
        function probeClickCB(obj, ~, keydata)
            if keydata.Button==1 % left click
                obj.P.currX = round(keydata.IntersectionPoint(1));
                obj.P.currY = round(keydata.IntersectionPoint(2));
            else % any other click, disconnect/reconnect the nearest channel
                thisX = round(keydata.IntersectionPoint(1));
                thisY = round(keydata.IntersectionPoint(2));
                xc = obj.P.chanMap.xcoords;
                yc = obj.P.chanMap.ycoords;
                dists = ((thisX-xc).^2+(thisY-yc).^2).^(0.5);
                [~,ii] = sort(dists);
                obj.P.chanMap.connected(ii(1)) = ~obj.P.chanMap.connected(ii(1));                
            end
            obj.updateProbeView;
            obj.updateDataView;
        end
        
        function dataClickCB(obj, ~, keydata)                   
            try
            if keydata.Button==1 % left click, re-center view
                obj.P.currT = keydata.IntersectionPoint(1)-diff(obj.P.tWin)/2;
                
                thisY = round(keydata.IntersectionPoint(2)); 
                if thisY<=0; thisY = 1; end
                if thisY>=numel(obj.P.selChans); thisY = numel(obj.P.selChans); end
                thisCh = obj.P.selChans(thisY);
                obj.P.currY = obj.P.chanMap.ycoords(thisCh);
                
            else % any other click, disconnect/reconnect the nearest channel   
                thisY = round(keydata.IntersectionPoint(2));  
                if thisY<=0; thisY = 1; end
                if obj.P.colormapMode
                    sc = obj.P.selChans;
                    cm = obj.P.chanMap;
                    sc = sc(ismember(sc, find(cm.connected)));
                    if thisY<=0; thisY = 1; end
                    if thisY>=numel(sc); thisY = numel(sc); end
                    thisCh = sc(thisY);                    
                else
                    if thisY>=numel(obj.P.selChans); thisY = numel(obj.P.selChans); end
                    thisCh = obj.P.selChans(thisY);                
                end
                obj.P.chanMap.connected(thisCh) = ~obj.P.chanMap.connected(thisCh);
            end
            obj.updateProbeView;
            obj.updateDataView;
            end
        end
        
        
        %% timeClickCB
        function timeClickCB(obj, ~, keydata)
            if obj.P.dataGood
                nSamp = obj.P.nSamp;
                maxT = nSamp/obj.ops.fs;
                
                obj.P.currT = keydata.IntersectionPoint(1)*maxT;
                set(obj.H.timeLine, 'XData', keydata.IntersectionPoint(1)*[1 1]);
                
                obj.updateDataView;
            end
            
        end
        
        
        %% timeClickLeftCB
        function timeClickLeftCB(obj, ~, keydata)
            if obj.P.dataGood
                nSamp = obj.P.nSamp;
                maxT = nSamp/obj.ops.fs;
                % step left (backward)
                newT = max(obj.P.currT-10, 0);
                tdiff = newT-obj.P.currT;
                if tdiff
                    % update current time marker
                    obj.P.currT = newT;
                    set(obj.H.timeLine, 'XData', get(obj.H.timeLine, 'XData')+tdiff/maxT); %keydata.IntersectionPoint(1)*[1 1]);
                    
                    obj.updateDataView;
                end
            end
        end %timeClickRightCB
        
        
        %% timeClickRightCB
        function timeClickRightCB(obj, ~, keydata)
            if obj.P.dataGood
                nSamp = obj.P.nSamp;
                maxT = nSamp/obj.ops.fs;
                % step right(forward) 10 sec
                newT = min(obj.P.currT+10, maxT);
                tdiff = newT-obj.P.currT;
                if tdiff
                    % update current time marker
                    obj.P.currT = newT;
                    set(obj.H.timeLine, 'XData', get(obj.H.timeLine, 'XData')+tdiff/maxT); %keydata.IntersectionPoint(1)*[1 1]);
                    
                    obj.updateDataView;
                end
            end
        end %timeClickRightCB
        
        
        function keyboardFcn(obj, ~, k)
            m = get(obj.H.fig,'CurrentModifier');

            if isempty(m)
                switch k.Key
                    case 'uparrow'
                        if ~obj.P.colormapMode
                            obj.P.nChanToPlot = obj.P.nChanToPlot+1;
                            if obj.P.nChanToPlot > numel(obj.P.chanMap.chanMap)
                                obj.P.nChanToPlot = numel(obj.P.chanMap.chanMap);
                            end
                        end
                    case 'downarrow'
                        if ~obj.P.colormapMode
                            obj.P.nChanToPlot = obj.P.nChanToPlot-1;
                            if obj.P.nChanToPlot == 0
                                obj.P.nChanToPlot = 1;
                            end
                        end
                    case 'c'
                        obj.P.colormapMode = ~obj.P.colormapMode;
                    case '1'
                        if obj.P.colormapMode
                            obj.P.showRaw = true;
                            obj.P.showWhitened = false;
                            obj.P.showPrediction = false;
                            obj.P.showResidual = false;
                        else
                            obj.P.showRaw = ~obj.P.showRaw;
                        end
                    case '2'
                        if obj.P.colormapMode
                            obj.P.showRaw = false;
                            obj.P.showWhitened = true;
                            obj.P.showPrediction = false;
                            obj.P.showResidual = false;
                        else
                            obj.P.showWhitened = ~obj.P.showWhitened;
                        end
                    case '3'
                        if obj.P.colormapMode
                            obj.P.showRaw = false;
                            obj.P.showWhitened = false;
                            obj.P.showPrediction = true;
                            obj.P.showResidual = false;
                        else
                            obj.P.showPrediction = ~obj.P.showPrediction;
                        end
                    case '4'
                        if obj.P.colormapMode
                            obj.P.showRaw = false;
                            obj.P.showWhitened = false;
                            obj.P.showPrediction = false;
                            obj.P.showResidual = true;
                        else
                            obj.P.showResidual = ~obj.P.showResidual;
                        end
                        
                end
            elseif strcmp(m, 'control')
                % don't break normal window focus shortcuts!
                % - could not find equivalent of Editor window shortcut (cmd-shift-0), but that will
                %   at least work from command window
                switch k.Key
                    case '0'                            
                        commandwindow
                end
            end

            obj.updateProbeView;
            obj.updateDataView;
        end
            
        
        function saveGUIsettings(obj)
            
            saveDat.settings.ChooseFileEdt.String = obj.H.settings.ChooseFileEdt.String;
            % obsolete:            saveDat.settings.ChooseTempdirEdt.String = obj.H.settings.ChooseTempdirEdt.String;
            saveDat.settings.setProbeEdt.String = obj.H.settings.setProbeEdt.String;
            saveDat.settings.setProbeEdt.Value = obj.H.settings.setProbeEdt.Value;
            saveDat.settings.setnChanEdt.String = obj.H.settings.setnChanEdt.String;
            saveDat.settings.setFsEdt.String = obj.H.settings.setFsEdt.String;
            saveDat.settings.setThEdt.String = obj.H.settings.setThEdt.String;
            saveDat.settings.setLambdaEdt.String = obj.H.settings.setLambdaEdt.String;
            saveDat.settings.setCcsplitEdt.String = obj.H.settings.setCcsplitEdt.String;
            saveDat.settings.setMinfrEdt.String = obj.H.settings.setMinfrEdt.String;
            
            saveDat.ops = obj.ops;
            saveDat.ops.gui = [];
            saveDat.rez = []; %obj.rez;  % no, rez is too big to include when just saving gui settings
            saveDat.rez.cProjPC = []; 
            saveDat.rez.cProj = []; 
            saveDat.rez.ops.gui = [];
            saveDat.P = obj.P;
            
            try
                savePath = [];
                if ~strcmp(obj.H.settings.ChooseFileEdt.String, '...')
                    [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);
                    savePath = fullfile(p, [fn '_ksSettings.mat']);
                    save(savePath, 'saveDat', '-v7.3');
                end
            catch
               fprintf('Error occurred when attempting to save kilosort GUI settings:  %s\n', savePath); 
            end
            %obj.refocus(obj.H.settings.saveBtn);
        end
        
        function restoreGUIsettings(obj)
            [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);            
            savePath = fullfile(p, [fn '_ksSettings.mat']);
            
            if 0% exist(savePath, 'file')
                obj.log('Restoring saved session...');
                
                load(savePath);
                
                obj.H.settings.ChooseFileEdt.String = saveDat.settings.ChooseFileEdt.String;
                %  obsolete:    obj.H.settings.ChooseTempdirEdt.String = saveDat.settings.ChooseTempdirEdt.String;
                obj.H.settings.setProbeEdt.String = saveDat.settings.setProbeEdt.String;
                obj.H.settings.setProbeEdt.Value = saveDat.settings.setProbeEdt.Value;
                obj.H.settings.setnChanEdt.String = saveDat.settings.setnChanEdt.String;
                obj.H.settings.setFsEdt.String = saveDat.settings.setFsEdt.String;
                obj.H.settings.setThEdt.String = saveDat.settings.setThEdt.String;
                obj.H.settings.setLambdaEdt.String = saveDat.settings.setLambdaEdt.String;
                obj.H.settings.setCcsplitEdt.String = saveDat.settings.setCcsplitEdt.String;
                obj.H.settings.setMinfrEdt.String = saveDat.settings.setMinfrEdt.String;
                
                obj.ops = saveDat.ops;
                obj.rez = saveDat.rez;
                obj.P = saveDat.P;
                
                obj.updateProbeView('new');
                obj.updateDataView;
            end
        end
            
        function writeScript(obj)
            % write a .m file script that the user can use later to run
            % directly, i.e. skipping the gui
            obj.log('Writing to script not yet implemented.');
            obj.refocus(obj.H.settings.writeBtn);
        end
        
        function help(obj)
            
            hstr = {'Welcome to Kilosort!',...                
                '',...
                '*** Troubleshooting ***', ...
                '1. Click ''reset'' to try to clear any GUI problems or weird errors. Also try restarting matlab.', ...                
                '2. Visit github.com/MouseLand/Kilosort2 to see more troubleshooting tips.',...
                '3. Create an issue at github.com/MouseLand/Kilosort2 with as much detail about the problem as possible.'};
            
            h = helpdlg(hstr, 'Kilosort help');
            
        end
        
        function reset(obj)
             % full reset: delete userSettings.mat and the settings file
             % for current file. re-launch. 
             try
                 if exist(obj.P.settingsPath)
                     delete(obj.P.settingsPath);
                 end
                 
                 [p,fn] = fileparts(obj.H.settings.ChooseFileEdt.String);
                 savePath = fullfile(p, [fn '_ksSettings.mat']);
                 if exist(savePath, 'file')
                     delete(savePath);
                 end
             end
             
             kilosort;
             
        end
        
        function cleanup(obj)
            if ~isfield(obj.P, 'saveSettingsOnClose') || obj.P.saveSettingsOnClose
                obj.saveGUIsettings();
            end
            fclose('all');
        end
        
        %% log
        function log(obj, message)
            % show a message to the user in the log box
            timestamp = datestr(now, 'dd-mm  HH:MM:SS');
            str = sprintf('[%s]\t  %s', timestamp, message);
            % If error, tint logBox red
            bgColor = [1 1 1] - [0 1 1]*.3*contains(str,'Error');
            current = get(obj.H.logBox, 'String');
            set(obj.H.logBox, 'String', [current; str], ...
                'Value', numel(current) + 1, 'BackgroundColor', bgColor);
            % Send everything to command window too
            fprintf('%s\n', str);
            drawnow;
        end
    end
    
    methods(Static)
        
        function refocus(uiObj)
            set(uiObj, 'Enable', 'off');
            drawnow update;
            set(uiObj, 'Enable', 'on');
        end
        
        function ops = defaultOps()
            % look for a default ops file and load it
%             if exist('defaultOps.mat')
%                 load('defaultOps.mat', 'ops');
            if exist('configFile384.m', 'file')
                configFile384;
                ops.chanMap     = [];
                ops.trange      = [0 Inf];
            else
                ops = [];
            end
        end
        
        function docString = opDoc(opName)
            switch opName
                case 'NchanTOT'; docString = 'Total number of rows in the data file';
                case 'Th'; docString = 'Threshold on projections (like in Kilosort1)';
            end
        end
        
        function isIn = isInLims(ax)
            cp = get(ax, 'CurrentPoint');
            xl = get(ax, 'XLim');
            yl = get(ax, 'YLim');
            
            isIn = cp(1)>xl(1) && cp(1)<xl(2) && cp(1,2)>yl(1) && cp(1,2)<yl(2);
        end
        
        function newRange = chooseNewRange(oldRange, scaleFactor,newCenter, maxRange)
            dRange = diff(oldRange);
            mn = maxRange(1); mx = maxRange(2);
            
            if newCenter>mx; newCenter = mx; end
            if newCenter<mn; newCenter = mn; end
            dRange = dRange*scaleFactor;
            
            if dRange>(mx-mn); dRange = mx-mn; end
            newRange = newCenter+[-0.5 0.5]*dRange;
            if newRange(1)<mn
                newRange = newRange+mn-newRange(1);
            elseif newRange(2)>mx
                newRange = newRange+mx-newRange(2);
            end
            
        end
        
    end
    
end


