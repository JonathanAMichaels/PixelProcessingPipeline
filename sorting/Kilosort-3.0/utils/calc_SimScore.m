function simScore = calc_SimScore(rez)

    % recalculate simScore including temporal lag
    % - cranky implementation due to differences in rez.W and wPCA dimensions
    %  - weird/deep inconsistencies & hardcoding of how many PCs are preserved (3, 6, 2*3?<!?)
    U = permute(rez.U, [2,1,3]);
    W = permute(rez.W, [2,1,3]);
    [Nfilt, nt0, nPC] = size(W);
    simScore = (U(:,:) * U(:,:)') .* (W(:,:) * W(:,:)')/nPC;
    wPCA  = gather(rez.wPCA(:,1:nPC));
    wroll = [];
    tlag = [-4:1:4];%[-2, -1, 1, 2];
    for j = 1:length(tlag)
        wroll(:,:,j) = circshift(wPCA, tlag(j), 1)' * wPCA;
    end
    
    for j = 1:size(wroll,3)
        Wr = reshape(W, [Nfilt * nt0, nPC]);
        Wr = Wr * wroll(:,:,j)';
        Wr = reshape(Wr, [Nfilt, nt0, nPC]);
        Xsim =  (U(:,:) * U(:,:)') .* (Wr(:,:) * W(:,:)')/nPC;
        simScore = max(simScore, Xsim);
    end
    simScore = gather(simScore);
    %rez.simScore = simScore;
end