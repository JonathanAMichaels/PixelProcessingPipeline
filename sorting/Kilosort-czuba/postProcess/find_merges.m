function rez = find_merges(rez, flag)
% this function merges clusters based on template correlation
% however, a merge is veto-ed if refractory period violations are introduced

ops = rez.ops;
dt = 1/1000;
ccgLB = 0.5; % lower bound for merge consideration (defaults: KS2==0.5, KS3==0.7)

if ~isfield(rez,'simScore') || isempty(rez.simScore)
    % recalculate simScore including temporal lag
    % - cranky implementation due to differences in rez.W and wPCA dimensions
    %  - weird/deep inconsistencies & hardcoding of how many PCs are preserved (3, 6, 2*3?<!?)
    %     U = permute(rez.U, [2,1,3]);
    %     W = permute(rez.W, [2,1,3]);
    %     [Nfilt, nt0, nPC] = size(W);
    %     simScore = (U(:,:) * U(:,:)') .* (W(:,:) * W(:,:)')/nPC;
    %     wPCA  = gather(rez.wPCA(:,1:nPC));
    %     wroll = [];
    %     tlag = [-4:1:4];%[-2, -1, 1, 2];
    %     for j = 1:length(tlag)
    %         wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
    %     end
    %
    %     for j = 1:size(wroll,3)
    %         Wr = reshape(W, [Nfilt * nt0, nPC]);
    %         Wr = Wr * wroll(:,:,j)';
    %         Wr = reshape(Wr, [Nfilt, nt0, nPC]);
    %         Xsim =  (U(:,:) * U(:,:)') .* (Wr(:,:) * W(:,:)')/nPC;
    %         simScore = max(simScore, Xsim);
    %     end
    rez.simScore = calc_SimScore(rez);
    
%     dmu = 2 * abs(rez.mu' - rez.mu ) ./ (rez.mu' + rez.mu);
    Xsim = rez.simScore; % .* (dmu < .2);
% else
%     % use existing simScore
%     Xsim = rez.simScore; % this is the pairwise similarity score
end

Xsim = rez.simScore;

Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim)); % remove the diagonal of ones


% sort by firing rate first
nspk = accumarray(rez.st3(:,2), 1, [Nk, 1], @sum);
[~, isort] = sort(nspk); % we traverse the set of neurons in ascending order of firing rates

fprintf('initialized spike counts\n')

if ~flag
  % if the flag is off, then no merges are performed
  % this function is then just used to compute cross- and auto- correlograms
   rez.R_CCG = Inf * ones(Nk);
   rez.Q_CCG = Inf * ones(Nk);
   rez.K_CCG = {};
end

for j = 1:Nk
    s1 = rez.st3(rez.st3(:,2)==isort(j), 1)/ops.fs; % find all spikes from this cluster
    if numel(s1)~=nspk(isort(j))
        fprintf('lost track of spike counts') %this is a check for myself to make sure new cluster are combined correctly into bigger clusters
    elseif numel(s1)<=25
        % too few spikes in this unit?!?
        continue
    end
    % sort all the pairs of this neuron, discarding any that have fewer spikes
    [ccsort, ix] = sort(Xsim(isort(j),:) .* (nspk'>numel(s1)), 'descend');
    %ienu = find(ccsort<ccgLB, 1) - 1; % find the first pair which has too low of a correlation
    thesePairs = gather(ix(ccsort>=ccgLB));
    
    % for all pairs above 0.5 correlation
    for k = thesePairs %1:ienu
        s2 = rez.st3(rez.st3(:,2)==k, 1)/ops.fs; % find the spikes of the pair
        % compute cross-correlograms, refractoriness scores (Qi and rir), and normalization for these scores
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
        Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
        R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes
        if isnan(Q), keyboard, end
        if flag
            if flag==2 || (Q<.2 && R<.05) % if both refractory criteria are met
                i = k;
                % now merge j into i and move on
                rez.st3(rez.st3(:,2)==isort(j),2) = i; % simply overwrite all the spikes of neuron j with i (i>j by construction)
                nspk(i) = nspk(i) + nspk(isort(j)); % update number of spikes for cluster i
                fprintf('merged %d into %d \t[Q%01.2f, R%01.3f)\n', isort(j), i, Q, R)
                % YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                break; % if a pair is found, we don't need to keep going (we'll revisit this cluster when we get to the merged cluster)
            end
        else
          % sometimes we just want to get the refractory scores and CCG
            rez.R_CCG(isort(j), k) = R;
            rez.Q_CCG(isort(j), k) = Q;

            rez.K_CCG{isort(j), k} = K;
            rez.K_CCG{k, isort(j)} = K(end:-1:1); % the CCG is "antisymmetrical"
        end
    end
end

if ~flag
    rez.R_CCG  = min(rez.R_CCG , rez.R_CCG'); % symmetrize the scores
    rez.Q_CCG  = min(rez.Q_CCG , rez.Q_CCG');
end
