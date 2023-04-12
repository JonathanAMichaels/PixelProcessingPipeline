% mexGPUall. For these to complete succesfully, you need to configure the
% Matlab GPU library first (see README files for platform-specific
% information)

enableStableMode = true;

mexcuda -largeArrayDims spikedetector3.cu
mexcuda -largeArrayDims mexThSpkPC.cu
mexcuda -largeArrayDims mexGetSpikes2.cu

if enableStableMode
    % For algorithm development purposes which require guaranteed
    % deterministic calculations, add -DENSURE_DETERM swtich to
    % compile line for mexMPnu8.cu. -DENABLE_STABLEMODE must also
    % be specified. This version will run ~2X slower than the
    % non deterministic version.
    mexcuda -largeArrayDims -dynamic -DENABLE_STABLEMODE mexMPnu8.cu
else
    mexcuda -largeArrayDims mexMPnu8.cu
end

mexcuda -largeArrayDims mexSVDsmall2.cu
mexcuda -largeArrayDims mexWtW2.cu
mexcuda -largeArrayDims mexFilterPCs.cu
mexcuda -largeArrayDims mexClustering2.cu
mexcuda -largeArrayDims mexDistances2.cu

%    mex -largeArrayDims mexMPmuFEAT.cu
%    mex -largeArrayDims mexMPregMU.cu
%    mex -largeArrayDims mexWtW2.cu

% If you get uninterpretable errors, run again with verbose option -v, i.e. mexcuda -v largeArrayDims mexGetSpikes2.cu

% mex additional "_pcTight" versions, which compute template & feature projections using a temporal window
% more tightly restricted around the spike time
% - tightening the projection window to exclude less informative samples in the tails of the waveform improves
%   specificity of the features without leaving tails in the raw data after subtraction (tails are still subtracted)
% - instead of using the whole waveform t= 0:nt0, these use t=6:nt0-15
% - with the default nt0 length of 61 samples, this corresponds to a waveform length 40 samples
%   (i.e. 1 usec waveform projection, instead of 1.5 usec)

fprintf('\nProcessing...')
cp = fileparts(mfilename('fullpath'));
fn = dir(fullfile(cp,'*pcTight.cu'));
fn = {fn.name};
for i = 1:length(fn)
    estr = '-largeArrayDims ';
    if contains(fn{i}, 'mexMPnu8')
        estr = [estr,'-dynamic -DENABLE_STABLEMODE '];
    end
    
    fprintf('\n\t%s...', fn{i});
    eval(sprintf('mexcuda %s%s', estr, fn{i}));
end
fprintf('\nDONE.\n')
    
