function ex_mpmop(test_funcs, total_tasks, ar_values, num_workers)
% ex_mpmop: Batch driver for the MPMOP-CMA algorithm.
%
%   Inputs:
%     test_funcs  : vector of function indices (default 1)
%     total_tasks : independent runs per function (default 1)
%     ar_values   : scalar or vector of resource-allocation ratios (default 0.2)
%     num_workers : number of parallel workers (default 1)
%
%   Example:
%      ex_mpmop                     – test (default): func 1, 2 run, ar=0.2, 2 workers
%      ex_mpmop(1:24, 30, [], 8)    – full benchmark, 30 runs, ar=0.2 (default), 8 workers
%
% Empty arguments fallback to their defaults.

% ---------- test settings ----------
if nargin < 1 || isempty(test_funcs),   test_funcs   = 1:1;  end
if nargin < 2 || isempty(total_tasks),  total_tasks  = 2;    end
if nargin < 3 || isempty(ar_values),    ar_values    = 0.2;  end
if nargin < 4 || isempty(num_workers),  num_workers  = 2;    end

% ---------- parallel pool ----------
pool = gcp('nocreate');
if isempty(pool) || pool.NumWorkers ~= num_workers
    if ~isempty(pool), delete(pool); end
    pool = parpool(num_workers);
end

% ---------- output directory ----------
base_dir = fullfile('.', 'MPMOP_CMA', 'result', 'test');
if ~exist(base_dir, 'dir'), mkdir(base_dir); end

for ar = ar_values
    pr_total = []; 
    ar_suffix = int32(ar * 10); % for file name

    for idx_func = 1:numel(test_funcs)
        func = test_funcs(idx_func);

        all_peak   = cell(total_tasks, 1);
        all_speak  = cell(total_tasks, 1);
        all_sallpk = cell(total_tasks, 1);

        parfor (task_id = 1:total_tasks, num_workers)
            [pk, spk, sallpk] = MPMOP_CMA(func, task_id, ar);
            all_peak{task_id}   = pk;
            all_speak{task_id}  = spk;
            all_sallpk{task_id} = sallpk;
        end

        % --- concatenate and compute PR ---
        peak      = cat(1, all_peak{:});
        result    = cat(2, all_speak{:});
        sallpeak  = all_sallpk{1};

        pr_total = [pr_total; (mean(result, 2).' / sallpeak(1))]; %#ok<AGROW>
        % disp(pr_total)

        % --- per-function files ---
        writematrix(peak,   fullfile(base_dir, sprintf('P%d_ar_%d.csv', func, ar_suffix)));
        writematrix(result, fullfile(base_dir, sprintf('F%d_ar_%d.csv', func, ar_suffix)));
    end

    % --- summary ar value ---
    writematrix(pr_total, fullfile(base_dir, sprintf('F_total_ar_%d.csv', ar_suffix)));
end

disp('All results have been successfully saved.');

delete(pool);
end
