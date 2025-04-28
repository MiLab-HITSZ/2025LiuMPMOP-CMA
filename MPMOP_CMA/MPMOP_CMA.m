function [peak, speak, sallpeak] = MPMOP_CMA(fn, run, ar)
% MPMOP_CMA: Single run of the MPMOP-CMA framework on one dynamic test problem.
%
%   Inputs:
%     fn   : index of the DMMOP benchmark function (1–24)
%     run  : independent run index, reused as the RNG seed (reproducible)
%     ar   : allocation ratio – share of remaining evaluations given to the
%            multiparty-multiobjective search stage in each environment
%
%   Outputs:
%     peak      : number of global peaks found in every environment
%     speak     : cumulative peaks per environment (row vector)
%     sallpeak  : total number of true global peaks (scalar)
%
%   NOTE: The algorithm follows the four-stage loop described in the paper:
%         1. Multiparty multiobjective search (MPSearch)
%         2. CMA-ES local search (CMASearch)
%         3. Additional search (AdSearch) when all CMA models terminate
%         4. Dynamic response strategies when the environment changes

    algRand = RandStream.create('mt19937ar', 'Seed', run);
    RandStream.setGlobalStream(algRand);

    pro = DMMOP(fn);
    NP = [250, 250, 250, 250, 500, 500, 100, 500*ones(1, 9), 300, 300, 300, 300, 500, 500, 500, 500];
    NP = NP(fn);
    D = pro.D;
    x = rand(NP, D) .* (pro.upper - pro.lower) + pro.lower;
    fit = -pro.GetFits(x);
    DM = 2;

    % ---------- algorithm parameters --------------
    tw       = 20;       % window length for history solutions
    rp       = 5;        % reproduce factor when re-initialising
    ur       = 0.5;      % upper ratio of history-derived individuals
    sol      = {};       % cell array storing history solution sets
    minorGen = 3;        % generations for additional search
	

    while ~pro.Terminate()
		fprintf('MPMOP-CMA runing, funcNo:%d, run index:%d, environment:%d\n', fn, run, pro.env+1);
        
        % -------- stage 1 : multiparty multiobjective search -------------
        rest   = pro.freq - rem(pro.evaluated, pro.freq);   % remaining evals
        maxGen = floor(rest / NP * ar);                     % # generations
        [x, fit] = MPSearch(x, fit, NP, maxGen, DM, pro, algRand);
        

        % -------- stage 2 : CMA-ES local search ----------------------
        S  = Initialize_CMA(x, fit, pro); 
        bx = cat(1, S.bx);
        bf = cat(1, S.bf);

        while ~pro.CheckChange(bx, -bf)
            ter = [S.ter];
            if all(ter)                                   % all CMA terminated
                % -------- stage 3 : additional search --------------------
                [x, fit, newS] = AdSearch(sol, x, fit, minorGen, ...
                                           DM, pro, algRand, rp, NP, ur);
                S = [S; newS];                            % append new models
            end
            
            ter = [S.ter];
            k = find(ter == false, 1);
            S = CMASearch(S, k, pro);
            bx = cat(1, S.bx);
            bf = cat(1, S.bf);
        end

        % -------- stage 4 : dynamic response -----------------------------
        valid_indices = [S.valid] == 1 & [S.cmaGen] >= 1;
        bx = cat(1, S(valid_indices).bx);
        bf = cat(1, S(valid_indices).bf);



        [~, rank] = sort(bf);
        bx  = bx(rank, :);
        sol = [sol, {bx}];
        if numel(sol) > tw, sol(1) = []; end

        addpop = reinit_history(sol, pro, rp, NP, ur, true);
        x = rand(NP, pro.D) .* (pro.upper - pro.lower) + pro.lower;
        x(1:size(addpop, 1), :) = addpop;
        fit = -pro.GetFits(x);
    end

    [peak, allpeak] = pro.GetPeak();
    speak = sum(peak, 2);
    sallpeak = sum(allpeak);
end
