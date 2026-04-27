# Resilience-Effort Control Synthesis

Python code for solving Resilience-Effort optimization problems with bounded disturbances using [Pyomo](http://www.pyomo.org/). Provides three optimizers:

- **`ParetoOptimizer`** — jointly trades off the control effort `epsilon` and the disturbance radius `mu` via a weighted scalar objective `w[0]*epsilon - w[1]*mu`.
- **`EffortOptimizer`** — minimizes the control effort needed `epsilon` for a fixed disturbance radius `r`.
- **`ResilienceOptimizer`** — dual of the effort formulation: fixes a control-effort level `epsilon` and **maximizes** the disturbance radius `mu` the system can tolerate.

Both use the same parameterized affine state-feedback control law:

```
u(k) = alpha_1 * x(k) + alpha_2
```

## Installation

Install dependencies and use the package locally:

```bash
pip install -r requirements.txt
```

### Solver

The default solver is **IPOPT** (`ipopt`). Other solver can be used depending on avalability and utility.

## Quick start

### Effort optimizer (fixed `r`, minimize `epsilon`)

```python
import numpy as np
from ResilienceEffort import EffortOptimizer, plot_closed_loop_trajectories

N, n, m, q = 6, 2, 2, 4
A = np.eye(2)
B = np.eye(2)

opt = EffortOptimizer(N=N, n=n, m=m, q=q, A=A, B=B)
G = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
H_safe = np.array([1.0, 1.7, 0.0, 2.0])
opt.set_constraints([G] * (N + 1), [H_safe] * (N + 1))
opt.set_disturbance_bound(r=0.05)

x0 = np.array([0.0, 0.2])
sol = opt.solve(x0, verbose=True)
# sol = {"epsilon": ..., "alpha_1": ..., "alpha_2": ..., "P": ...}

plot_closed_loop_trajectories(
    opt, x0,
    b1=[-0.3, 0.3, 0.6, 1.25],
    b2=[0.8, 1.5, 1.2, 1.75],
    safe_bounds=[-1.0, 1.7, 0.0, 2.0],
)
```


## CLI

The package ships with a small CLI driven by a JSON config file. Choose between the three optimizers with `--mode`:

```bash
# Effort mode (fixed mu, minimize epsilon)
python -m ResilienceEffort.cli --config examples/config.json --plot --verbose

# Pareto mode
python -m ResilienceEffort.cli --config examples/pareto_config.json --mode pareto --plot

# Resilience mode (fixed epsilon, maximize mu)
python -m ResilienceEffort.cli --config examples/resilience_config.json --mode resilience --solver ipopt --plot

# Override individual parameters
python -m ResilienceEffort.cli --config examples/config.json --r 0.05 --N 8 --plot
python -m ResilienceEffort.cli --config examples/pareto_config.json --mode pareto --w 0.05 0.2 --plot
python -m ResilienceEffort.cli --config examples/resilience_config.json --mode resilience --epsilon 1.2 --plot
```


Mode-specific fields:
- **Effort mode**: add `"r": 0.068` (disturbance bound).
- **Pareto mode**: add `"w": [0.02, 0.1]` (weights `[w_epsilon, w_mu]`).
- **Resilience mode**: add `"epsilon": 0.93` (control-effort budget).


## Project layout

```
robust-control/
├── ResilienceEffort/
│   ├── __init__.py
│   ├── optimizer.py            # _BaseOptimizer + Effort/Pareto/Resilience optimizers
│   ├── plotting.py             # plot_closed_loop_trajectories, plot_pareto_frontier
│   └── cli.py                  # CLI with --mode effort|pareto|resilience
├── examples/
│   ├── run_effort_example.py              # effort demo
│   ├── run_pareto_example.py       # pareto demo
│   ├── run_resilience_example.py   # resilience demo
│   ├── run_pareto_curve.py         # Pareto frontier sweep + plot
│   ├── config.json                 # effort config
│   ├── pareto_config.json          # pareto config
│   └── resilience_config.json      # resilience config
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```


## Citation

If you use this code in your research, please cite:

> Si, Youssef Ait, et al. *Resilient and Effort-Optimal Controller Synthesis under Temporal Logic Specifications.* arXiv preprint arXiv:2604.10680 (2026).

BibTeX:

@article{si2026resilient,
  title={Resilient and Effort-Optimal Controller Synthesis under Temporal Logic Specifications},
  author={  AitSi, Youssef and Das, Ratnangshu and Monir, Negar and Soudjani, Sadegh and Jagtap, Pushpak and Saoud, Adnane},
  journal={arXiv preprint arXiv:2604.10680},
  year={2026}}

