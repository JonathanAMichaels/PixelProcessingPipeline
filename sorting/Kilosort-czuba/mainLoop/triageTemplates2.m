function [W, U, dWU, mu, nsp, ndrop filterAge] = ...
    triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop, filterAge)
% This function checks if some templates should be dropped
% either because they are very similar to another template,
% or because they are not catching any spikes, (low mean firing rate).
% Takes as inputs almost all the information that determines templates, and
% outputs the same variables back after removing some clusters.
%
% [ks25] updates:
% - had to appended [filterAge] input/output, for consistent template dropping
% - now dropping any templates with predominantly negative weighting
%   - instance count included in ndrop(1); drops due to low firing rate
% 
% ---
% 202x-xx-xx  TBC  Evolved from original Kilosort
% 2021-04-28  TBC  Cleaned & commented


% this is the firing rate threshold
m0 = ops.minFR * ops.NT/ops.fs;
idrop = nsp<m0; % drop any templates with firing rate below this

% also ditch any peak-negative or mean-negative weighted templates
dU = double(gather(U(:,:,1)));
[~,ui] = max(abs(dU),[], 'linear');
idrop = idrop | (dU(ui)<0)' | (mean(dU)<0)'; %(mean(double(gather(U(:,:,1))))'<0);

W(:,idrop,:) = []; % remove those templates everywhere
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
filterAge(idrop) = [];
ndrop(1) = .9 * ndrop(1) + .1*gather(sum(idrop)); % keep track of how many templates have been removed this way

% compute pairwise correlations between templates
cc = getMeWtW2(W, U);
cc = cc -diag(diag(cc)); % exclude the diagonal

sd = sqrt(10); % this is hard-coded here    % ~!~ why would this be hardcoded to a fixed value??
% sd = sqrt(mu); % this is hard-coded here

% compute a score for the separation of the means
r0 = 2*(sd(:) + sd(:)') ./ abs(mu(:) - mu(:)');
r0(eye(size(r0),'logical')) = 0;    % exclude the diagonal
% determine which template has more spikes (that one survives)
rdir = (nsp(:) - nsp(:)')<0;
% for each pair of template, score their similarity by their template correlation, and amplitude separation
ipair = (cc>0.9 & r0>1 & rdir);
% for each template, find its most similar other template
amax = max(ipair, [], 2);
% if this score is 1, then all the criteria have been met for dropping this template
idrop= amax>0;

% remove these templates everywhere like before
W(:,idrop,:) = [];
U(:,idrop,:) = [];
dWU(:,:, idrop) = [];
mu(idrop) = [];
nsp(idrop) = [];
filterAge(idrop) = [];
ndrop(2) = .9*ndrop(2) + .1*gather(sum(idrop)); % keep track of how many templates have been removed this way

% check which templates can be absorbed into other templates

% mergeThreshold = getOr(ops, 'mergeThreshold', 0);
%
% imerge = find(sqrt(dnext(:)) < mergeThreshold * mu);
% iW0 = iW(imerge);
% nsp0 = nsp(imerge);
%
% p2p = (C2C(iW0, iW0)<60) .* (nsp0(:) > nsp0(:)');
% imax = max(p2p, [], 2) < .5;
%
% idrop = false(numel(mu), 1);
% idrop(imerge(imax)) = 1;
%
% W(:,idrop,:) = [];
% U(:,idrop,:) = [];
% dWU(:,:, idrop) = [];
% mu(idrop) = [];
% nsp(idrop) = [];
% sig(idrop) = [];
% dnext(idrop) = [];
%
% damp(idrop, :) = [];
% ndrop(3) = .9*ndrop(3) + .1*gather(sum(idrop));
%
