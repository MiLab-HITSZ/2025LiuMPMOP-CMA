function Rank = MPRank(x, PopObj, DM)
% MPRank: Pareto ranking for multiparty optimization.
%
%   Inputs:
%     x       : N × D decision matrix
%     PopObj  : N x (2*DM) objective matrix (two objeWctives)
%     DM      : number of decision makers
%
%   Output:
%     Rank    : N × DM matrix, each column is the Pareto level under one DM

N          = size(PopObj, 1);
numGroups  = DM;

% -------------------------------------------------------------------------
% 1) Neighbourhood threshold d
% -------------------------------------------------------------------------
distMat               = pdist2(x, x);      % full pairwise distance
distMat(eye(N) == 1)  = inf;               % ignore self–distance
mindis  = mean(min(distMat));              % average nearest distance
d       = 2.0 * mindis;                    % neighbourhood radius
isNbr   = distMat < d;                     % logical neighbour mask

% -------------------------------------------------------------------------
% 2) Build dominance matrices – evaluate only neighbour pairs
% -------------------------------------------------------------------------
DominateCell = cell(1, numGroups);
for j = 1:numGroups
    DominateCell{j} = false(N);
end

for i = 1:N-1
    nbrIdx = find(isNbr(i, i+1:N)) + i;    % neighbours
    if isempty(nbrIdx), continue; end
    for j = 1:numGroups
        cols   = (2*j-1):(2*j);
        obj_i  = PopObj(i,        cols);
        obj_n  = PopObj(nbrIdx,   cols);

        less    = any(bsxfun(@lt, obj_i, obj_n), 2);
        greater = any(bsxfun(@gt, obj_i, obj_n), 2);

        DominateCell{j}(i,  nbrIdx( less  & ~greater)) = true;
        DominateCell{j}(nbrIdx(~less &  greater),  i)  = true;
    end
end

% -------------------------------------------------------------------------
% 3) Pareto front peeling
% -------------------------------------------------------------------------
RankCell  = cell(1, numGroups);
BeDom_np  = zeros(numGroups, N);

for j = 1:numGroups
    RankCell{j} = zeros(N, 1);
end

for i = 1:N
    for j = 1:numGroups
        BeDom_np(j,i) = sum(DominateCell{j}(:, i));
        if BeDom_np(j,i) == 0
            RankCell{j}(i) = 1;
        end
    end
end

for j = 1:numGroups
    level = 1;
    while any(RankCell{j} == 0)
        current = find(RankCell{j} == level)';
        for k = current
            dominated = find(DominateCell{j}(k, :));
            BeDom_np(j, dominated) = BeDom_np(j, dominated) - 1;
            newly = BeDom_np(j, dominated) == 0;
            RankCell{j}(dominated(newly)) = level + 1;
        end
        level = level + 1;
    end
end

Rank = cell2mat(RankCell);
end
