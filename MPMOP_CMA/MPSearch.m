function [x, fit, FrontNo] = MPSearch(x, fit, NP, maxGen, DM, pro, algRand)
% MPSearch  Multiparty multi-objective search.
%
%   Inputs:
%     x, fit  : current pop and fitness
%     NP      : pop size
%     maxGen  : #generations
%     DM      : number of decision makers
%     pro     : problem object
%     algRand : RandStream from caller
%
%   Output:
%     x, fit  : updated population (size NP)
%     FrontNo : Pareto-front index of the retained individuals

    curGen = 0;
    minRefer = min(fit);
    maxRefer = max(fit);

    while curGen < maxGen
        %% Differential Evolution (rand/1/bin)
        F  = 0.5;                        % scale factor
        CR = 0.7;                        % crossover probability
        offsX   = DE(x, NP, F, CR, pro, algRand);
        offsFit = -pro.GetFits(offsX);
        
        if pro.change
            return
        end
        
        % Update global min / max references
        minRefer = min(minRefer, min(offsFit));
        maxRefer = max(maxRefer, max(offsFit));
        
        % Combine parents and offspring
        x = [x; offsX];       %#ok<AGROW>
        fit = [fit; offsFit]; %#ok<AGROW>
        
        %% Global non-dominated sorting
        % Objective mapping and local ranking
        mFit = MPFit(x, fit, pro, DM, minRefer, maxRefer);
        rank = MPRank(x, mFit, DM);

         if DM == 1
            FrontNo = inf(size(x, 1), 1);
            idx = [];
            r = 1;
            MaxFNo = 1;
            while(length(idx) < NP)
                MaxFNo = r;
                temp = find(rank == r);
                FrontNo(temp) = r;
                idx = [idx; temp];      %#ok<AGROW>
                r = r+1;
            end
        else
            [FrontNo,MaxFNo] = NDSort(rank,NP);
        end
        
        %% Crowding-distance truncation
        CrowdDis = CrowdingDistance(mFit,FrontNo);

        Next = FrontNo < MaxFNo; 
        Last     = find(FrontNo==MaxFNo);
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:NP-sum(Next)))) = true;
        
        x = x(Next,:);
        fit = fit(Next, :);
        FrontNo = FrontNo(Next);
        curGen = curGen + 1;
    end
end
