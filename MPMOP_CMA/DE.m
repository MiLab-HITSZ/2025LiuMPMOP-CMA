function offsX = DE(x, N, F, CR, pro, algRand)
% DE: Differential Evolution operator (rand/1/bin).
%
%   Inputs:
%     x        : N × D parent matrix
%     N        : population size
%     F        : scale factor
%     CR       : crossover rate
%     pro      : problem object
%     algRand  : RandStream from caller

D = pro.D;

% 1) Mutation  (rand/1)
index = zeros(N, 3);
for i = 1:N
    index(i,:) = randperm(algRand, N-1, 3); % pick from {1..N} \ {i}
    index(i, index(i,:) >= i) = index(i, index(i,:) >= i) + 1;
end
mutant = x(index(:,1), :) - ...
         F .* (x(index(:,2), :) - x(index(:,3), :));

% 2) Binomial crossover
cross = rand(algRand, N, D) < CR;
parent_only = find(sum(cross,2) == 0);
for i = parent_only
    cross(i, randi(algRand, D)) = true;
end
offsX = cross .* mutant + (1-cross) .* x;

offsX = boundary_check(offsX, pro.lower, pro.upper);
end
