# 2025LiuMPMOP-CMA

This repository contains a fully-annotated MATLAB implementation of the algorithm proposed in "Multiparty Multiobjective Optimization for Dynamic Multimodal Optimization Problems".

## Directory layout

```
2025LiuMPMOP-CMA
├── MPMOP_CMA/           % core MATLAB source (algorithm + utilities)
│   ├── ex_mpmop.m           % batch driver for the benchmark
│   ├── MPMOP_CMA.m          % main algorithm
│   ├── MPSearch.m           % multiparty search
│   ├── CMASearch.m          % CMA-ES local search
│   ├── Initialize_CMA.m     % seed detection & CMA model init
│   ├── AdSearch.m           % additional search
│   ├── MPFit.m              % objective mapping
│   ├── MPRank.m             % local Pareto ranking
│   ├── DE.m, UpdateCMA.m, CrowdingDistance.m, NDSort.m, boundary_check.m
│   └── nbc_seeds.m          % two-level NBC niching
│
├── problem/              % CEC 2022 DMMOPs benchmark
│   ├── DMMOP.m
│   ├── dynamic_change.m, get_info.m, get_position.m
│   ├── Sub_f/ (F* test functions)   phi.mat, optima.mat, Data/*.mat
│
├── pyscript/             % (optional) Python demos
│   └── mp_simulation.py      % Simulation for Table II (Sec. III-A)
│
└── README.md            % this file
```

## Quick start (MATLAB R2020b+)

```matlab
% (Recommended) open MATLAB from the repo root so relative paths work
addpath(genpath('MPMOP_CMA'));
addpath(genpath('problem'));

% Example 1 – test, run a single function (P1), single run, ar=0.2, single core
ex_mpmop(1, 1, 0.2, 1);

% Example 2 – full benchmark (24 functions x 30 runs) using 8 workers
ex_mpmop(1:24, 30, 0.2, 8);
```

*Result files* are written to `./MPMOP_CMA/result/test/`:

| File                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| `P<i>_ar_<a>.csv`    | peaks found per environment for function *i*     |
| `F<i>_ar_<a>.csv`    | raw peak-found matrix                            |
| `F_total_ar_<a>.csv` | PR summary over all functions (one row per func) |

`<a>` is $10×ar$ (e.g. 2 for $ar = 0.2$).

> **Reproducibility** – each run index is reused as the RNG seed (`mt19937ar`), repeated executions produce identical CSVs.

## Python demo

The simulation script that analyses the dominance breaking probability in Section III-A is self-contained:

```bash
cd pyscript
python mp_simulation.py
```

It prints the conditional probability for several $(K1, K2)$ pairs.
