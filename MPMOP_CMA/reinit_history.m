function pop = reinit_history(archive, pro, rp, NP, ur, change)
% reinit_history: Generate new individuals from archive when the environment changes.
%
%   Inputs:
%     archive : cell array, each cell = population of one past environment
%     pro     : problem object
%     rp      : reproduce factor for each generated individual
%     NP      : pop size
%     ur      : upper ratio of history-based individuals (0–1)
%     change  : flag – true  means environment just changed
%
%   Output:
%     pop     : matrix of reinitialised individuals
%
% NOTES
%   Four strategies S1–S4 in the paper are implemented:
%     x1 – last solutions of each chain
%     x2 – linear extrapolation from two latest solutions
%     x3 – global bests over history chains
%     x4 – solutions not in any chain (unlinked)

    persistent data  linktable hisX
    
    if isempty(archive)
        pop = [];
        return
    end

    % 1) (re)build history data when environment just changed
    if change
        hisX = [];
        [data, linktable] = history_data(archive);
        
        for i = 1: length(data)
            tempX = data{i}(:, 1:end-1)';
            tempFit = -pro.GetFits(tempX);
            if pro.change, pop = []; return; end
            [~, minIdx] = min(tempFit);
            hisX = [hisX; tempX(minIdx, :)];  %#ok<AGROW>
        end
    end
        

    % 2) Strategy S2 : extrapolation from two latest solutions
    x2 = [];
    if size(data{1}, 2) > 1
        diffF = 0.5;
        xt1 = cell2mat(cellfun(@(c) c(:, end)',   data, 'UniformOutput', false));
        xt2 = cell2mat(cellfun(@(c) c(:, end-1)', data, 'UniformOutput', false));
        x2  = xt1 + diffF .* (xt1 - xt2);         % x_{t+1} prediction
        x2  = boundary_check(x2, pro.lower, pro.upper);
    end
    
    % 3) Strategy S1 : last solutions of each chain
    x1 = cell2mat(cellfun(@(x) x(:, end)', data, 'UniformOutput', false));
    x1 = boundary_check(x1, pro.lower, pro.upper);
    
    % 4) Strategy S3 : best solutions stored in hisX
    x3 = hisX;

    % 5) Strategy S4 : unlinked solutions in the previous archives
    x4 = [];
    for i = length(linktable):-1: 1
        curArchive = archive{i};
        curIdx = 1: size(curArchive, 1);
        curUnlinkIdx = setdiff(curIdx, linktable{i});
        if ~isempty(curUnlinkIdx)
            x4 = [x4; curArchive(curUnlinkIdx, :)];  %#ok<AGROW>
        end
        
    end
    
    % 6) Combine strategies, add Gaussian perturbations
    R = [x1; x2; x3; x4];
    % R = [x1; x2];

    variance_mat = 0.01 * eye(pro.D);    
    offsets = mvnrnd(zeros(1, pro.D), variance_mat, size(R, 1) * rp);

    pop = repmat(R, rp, 1) + offsets;
    pop = boundary_check(pop, pro.lower, pro.upper);
    pop = pop(1:min(size(pop, 1), ceil(NP * ur)), :);
end


function [current_history, linktable] = history_data(Archive)
% history_data: Build time chains and link table from Archive.
%
%   Outputs:
%     current_history : cell array, each cell = full time chain (TxD+1)'
%     linktable       : cell array, indices in each environment that belong
%                       to some chain (used to identify unlinked solutions)

current_archive = Archive{end};
num_solutions   = size(current_archive, 1);

current_history = cell(num_solutions, 1);
linktable       = cell(length(Archive), 1);
linktable{end}  = 1:num_solutions;            % init last env links

for m = 1:num_solutions
    cur_sol = current_archive(m, :);
    history = cur_sol;                        % start chain with latest sol

    % backwards through environments
    for t = length(Archive)-1 : -1 : 1
        prev_arc   = Archive{t};
        distances  = sqrt(sum((prev_arc - cur_sol).^2, 2));
        [~, idx]   = min(distances);
        history    = [prev_arc(idx, :); history]; %#ok<AGROW>
        cur_sol    = prev_arc(idx, :);
        linktable{t} = [linktable{t}, idx];
    end
    current_history{m} = history';
end
end
