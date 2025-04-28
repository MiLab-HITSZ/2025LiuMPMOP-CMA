function Sigma = CMASearch(Sigma, k, pro)
% CMASearch: One step for the k-th CMA-ES model.
%
%   Inputs:
%     Sigma : struct array, each entry is a CMA-ES model
%     k     : index of the active model to update
%     pro   : problem object
%
%   Output:
%     Sigma : updated struct array (entry k is modified)

ter   = [Sigma.ter];                % termination flags
tidx  = find(ter == true);          % indices of terminated models

bx = cat(1, Sigma.bx);              % historical best positions
bf = cat(1, Sigma.bf);              % best fitnesses

bxter = bx(tidx,:);                 % best positions of terminated models
bfter = bf(tidx);                   % best fitness   of terminated models


bxdis = pdist2(Sigma(k).bx, bxter); % distance to every terminated model
[minbxdis, minidx] = min(bxdis);    % closest terminated model
m = tidx(minidx);                   % its global index in Sigma

% no sampling points left
if isempty(Sigma(k).x)
    Sigma(k).ter = 1;
    return
end

% too close to a better terminated model
if ~isempty(tidx) && (minbxdis < 0.01 && bfter(m) < Sigma(k).bf)
     Sigma(k).valid = 0;
     Sigma(k).ter = 1;
     return
end

mu = 4 + floor(3 * log(pro.D));
OffsX = mvnrnd(Sigma(k).x, Sigma(k).sigma^2 * Sigma(k).C, mu);
OffsX = boundary_check(OffsX, pro.lower, pro.upper);
OffsFit = -pro.GetFits(OffsX);

if pro.change
    return
end

% update and record best-ever
[~, rank] = sort(OffsFit);
Sigma(k) = UpdateCMA(OffsX(rank, :), Sigma(k));

combineOffsFit = [OffsFit; Sigma(k).bf];
combineOffsX = [OffsX; Sigma(k).bx];
[minbf, minIdx] = min(combineOffsFit);

Sigma(k).bf = minbf;
Sigma(k).bx = combineOffsX(minIdx, :);
end