function S = Initialize_CMA(x, fit, pro)
% Initialize_CMA: Create a set of CMA-ES models from seed solutions.
%
%   Inputs:
%     x   : N × D decision matrix (population)
%     fit : N × 1 fitness vector (lower is better)
%     pro : problem object, provides dimension D
%
%   Output:
%     S   : struct array, each entry is an independent CMA-ES model

[~, idx] = sort(fit);
x   = x(idx, :);
fit = fit(idx);

seeds  = nbc_seeds(x, fit);
seedX  = x(seeds, :);

sk     = 1:length(seeds);
fk     = fit(seeds);

% CMA parameter initialize
S = struct( ...
    's',      num2cell(sk)', ...          % id
    'x',      num2cell(seedX, 2), ...     % current mean
    'sigma',  0.5, ...                    % step size
    'C',      eye(pro.D), ...             % covariance matrix
    'pc',     0, ...                      % covariance path
    'ps',     0, ...                      % step-size path
    'bx',     num2cell(seedX, 2), ...     % best position
    'bf',     num2cell(fk), ...           % best fitness
    'valid',  1, ...                      % valid flag
    'cmaGen', 0, ...                      % generations evolved
    'ter',    0);                         % termination flag

fliterd = [S.bf] <= mean(fit);
S = S(fliterd);

end