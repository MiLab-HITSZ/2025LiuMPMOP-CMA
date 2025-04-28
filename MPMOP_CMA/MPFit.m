function mFit = MPFit(x, fit, pro, DM, minRefer, maxRefer)
% MPFit: Build the 2xDM objectives (alpha+beta , 1-alpha+beta) for DMs.
%
% Inputs:
%   x        : N × D decision matrix
%   fit      : N × 1 raw fitness (single-objective)
%   pro      : problem object
%   DM       : number of decision makers
%   minRefer : best fitness seen
%   maxRefer : worst fitness seen
%
% Output:
%   mFit     : N × (2*DM) matrix of transformed objectives
%              column 2k-1 =  alpha_k(x) + beta(x)
%              column 2k   = 1-alpha_k(x) + beta(x)

normFit = (fit - minRefer) / (maxRefer - minRefer);
normFit = normFit(:);

groups = dimension_group(pro.D, DM);
N      = size(x, 1);
alpha  = zeros(N, DM);

normDec = (x - pro.lower) ./ (pro.upper - pro.lower);

for k = 1:DM
    alpha(:, k) = mean(normDec(:, groups{k}), 2);
end

mFit = zeros(N, 2*DM);

for k = 1:DM
    mFit(:, 2*k-1) = alpha(:, k) + normFit;              % MP_{k,1}
    mFit(:, 2*k  ) = 1 - alpha(:, k) + normFit;          % MP_{k,2}
end
end


function groups = dimension_group(D, K)
% dimension_group  Evenly partition 1:D into K contiguous index sets.
%
% Example: D = 7, K = 3  -->  {1:3, 4:5, 6:7}

base   = floor(D / K); 
remain = rem(D, K);
groups = cell(1, K);

idx = 1;
for k = 1:K
    extra = k <= remain;
    groups{k} = idx : idx + base + extra - 1;
    idx = idx + base + extra;
end
end