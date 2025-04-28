function [x, fit, newS] = AdSearch(sol, x, fit, minorGen, DM, pro, algRand, rp, NP, ur)
% AdSearch: Additional search stage after all CMA models have terminated.
%
%   Inputs:
%     sol       : history solution chains
%     x, fit    : current pop and fitness
%     minorGen  : #generations for the lightweight MPSearch
%     DM        : number of decision makers
%     pro       : problem object
%     algRand   : RandStream used by the caller
%     rp        : reproduce factor
%     NP        : population size upper bound
%     ur        : max ratio of history-derived individuals (0–1)
%
%   Outputs:
%     x, fit : updated population
%     newS   : initialized CMA-ES models

if ~isempty(sol)
    addpop = reinit_history(sol, pro, rp, NP, ur, false);
    addfit = -pro.GetFits(addpop);
    [addpop, addfit] = MPSearch(addpop, addfit, size(addpop, 1), minorGen, DM, pro, algRand);

    x = [x; addpop];
    fit = [fit; addfit];
    newS = Initialize_CMA(x, fit, pro);
    [~, rank] = sort(fit);
    x = x(rank(1:NP), :);
    fit = fit(rank(1:NP));
else
    [x, fit] = MPSearch(x, fit, size(x, 1), minorGen, DM, pro, algRand);
    newS = Initialize_CMA(x, fit, pro);
end

end

