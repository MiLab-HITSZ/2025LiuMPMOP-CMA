function nbcseeds = nbc_seeds(pop, ~)
%  NBC divides population into multi sub-popultions
%  For more information please refer to the
%  M. Preuss. "Niching the CMA-ES via nearest-better clustering." In
%  Proceedings of the 12th annual conference companion on Genetic and evolutionary
%  computation (GECCO ��10). ACM, New York, NY, USA, pp. 1711-1718, 2010.

% Calculate distance matrix
matdis = pdist2(pop, pop);

% Set parameter
factor = 2.0;

% Initialize NBC array
n = length(matdis);
nbc = zeros(n, 3);
nbc(:, 1) = 1:n;
nbc(1, 2) = -1;
nbc(1, 3) = 0;

% Calculate nearest better index and distance
for i = 2:n
    [u, v] = min(matdis(i, 1:i-1));
    nbc(i, 2) = v;
    nbc(i, 3) = u;
end

% Calculate mean distance and prune
meandis = factor * mean(nbc(2:n, 3));
nbc(nbc(:, 3) > meandis, 2) = -1;
nbc(nbc(:, 3) > meandis, 3) = 0;

% Find root with path compression
m = zeros(n, 2);
m(:, 1) = 1:n;
for i = 1:n
    m(i, 2) = find_root(i, nbc);
end

% Find seeds
seeds = unique(m(nbc(:, 2) == -1, 2));
nbcseeds = zeros(n, 1); % pre-allocate space
count = 0;

for i = 1:length(seeds)
    subidx = m(:, 2) == seeds(i);
    submeandis = factor * mean(nbc(subidx, 3));
    subseeds = find(subidx & nbc(:, 3) > submeandis);
    
    nbcseeds(count + 1) = seeds(i);
    count = count + 1;
    num_subseeds = length(subseeds);
    nbcseeds(count + 1:count + num_subseeds) = subseeds;
    count = count + num_subseeds;
end

% remove extra zeros
nbcseeds = nbcseeds(1:count);

nbcseeds = sort(nbcseeds);
end
% function to find root with path compression
function root = find_root(i, nbc)
    root = i;
    while nbc(root, 2) ~= -1
        root = nbc(root, 2);
    end
    while i ~= root
        parent = nbc(i, 2);
        nbc(i, 2) = root;
        i = parent;
    end
end