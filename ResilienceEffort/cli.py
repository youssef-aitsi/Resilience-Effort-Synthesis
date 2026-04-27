"""
Command-line interface for running the optimizers with custom parameters.

Two modes are supported:

* ``--mode effort`` (default): minimize ``epsilon`` for a fixed ``r``.
  Requires ``"r"`` in the config.
* ``--mode pareto``: jointly optimize ``epsilon`` and ``mu`` with weights.
  Requires ``"w"`` (length-2 list) in the config.

Examples
--------

    python -m ResilienceEffort.cli --config examples/config.json --plot
    python -m ResilienceEffort.cli --config examples/pareto_config.json --mode pareto --plot
    python -m ResilienceEffort.cli --config examples/config.json --r 0.05 --N 8 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .optimizer import ParetoOptimizer, ResilienceOptimizer, EffortOptimizer
from .plotting import plot_closed_loop_trajectories


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #
def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _bounds_to_H(b):
    b = np.asarray(b, dtype=float)
    return np.array([-b[0], b[1], -b[2], b[3]])


def _apply_common_config(optimizer, cfg: Dict[str, Any]) -> None:
    """Set G, H lists based on safe_bounds and per-step region_bounds."""
    G = np.array(cfg["G"], dtype=float)
    safe_H = _bounds_to_H(cfg["safe_bounds"])
    N = optimizer.N

    G_list = [G] * (N + 1)
    H_list = [safe_H] * (N + 1)

    region_bounds = cfg.get("region_bounds", {})
    for k_str, bounds in region_bounds.items():
        k = int(k_str)
        if not (0 <= k <= N):
            raise ValueError(f"region_bounds key {k} out of range [0, {N}]")
        H_list[k] = _bounds_to_H(bounds)

    optimizer.set_constraints(G_list, H_list)


def build_effort_optimizer(cfg: Dict[str, Any]) -> EffortOptimizer:
    """Build a :class:`EffortControlOptimizer` from a config dict."""
    optimizer = EffortOptimizer(
        N=int(cfg["N"]),
        n=int(cfg["n"]),
        m=int(cfg["m"]),
        q=int(cfg["q"]),
        A=np.array(cfg["A"], dtype=float),
        B=np.array(cfg["B"], dtype=float),
    )
    _apply_common_config(optimizer, cfg)
    if "r" not in cfg:
        raise KeyError("Effort mode requires 'r' (disturbance bound) in config.")
    optimizer.set_disturbance_bound(float(cfg["r"]))
    return optimizer


def build_pareto_optimizer(cfg: Dict[str, Any]) -> ParetoOptimizer:
    """Build a :class:`ParetoOptimizer` from a config dict."""
    optimizer = ParetoOptimizer(
        N=int(cfg["N"]),
        n=int(cfg["n"]),
        m=int(cfg["m"]),
        q=int(cfg["q"]),
        A=np.array(cfg["A"], dtype=float),
        B=np.array(cfg["B"], dtype=float),
    )
    _apply_common_config(optimizer, cfg)
    if "w" not in cfg:
        raise KeyError("Pareto mode requires 'w' (length-2 weight list) in config.")
    optimizer.set_weights(cfg["w"])
    return optimizer


def build_resilience_optimizer(cfg: Dict[str, Any]) -> ResilienceOptimizer:
    """Build a :class:`ResilienceOptimizer` from a config dict."""
    optimizer = ResilienceOptimizer(
        N=int(cfg["N"]),
        n=int(cfg["n"]),
        m=int(cfg["m"]),
        q=int(cfg["q"]),
        A=np.array(cfg["A"], dtype=float),
        B=np.array(cfg["B"], dtype=float),
    )
    _apply_common_config(optimizer, cfg)
    if "epsilon" not in cfg:
        raise KeyError("Resilience mode requires 'epsilon' (control-effort budget) in config.")
    optimizer.set_input_budget(float(cfg["epsilon"]))
    return optimizer


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the effort or Pareto MPC optimizer with user-defined parameters."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to a JSON configuration file (see examples/).",
    )
    parser.add_argument(
        "--mode", choices=["effort", "pareto", "resilience"], default="effort",
        help="Which optimizer to run (default: effort).",
    )
    parser.add_argument(
        "--solver", type=str, default="knitroampl",
        help="Pyomo solver to use (default: knitroampl).",
    )
    parser.add_argument("--N", type=int, default=None, help="Override prediction horizon.")
    parser.add_argument("--r", type=float, default=None, help="Override disturbance bound (effort mode).")
    parser.add_argument(
        "--w", type=float, nargs=2, default=None, metavar=("W_EPS", "W_MU"),
        help="Override Pareto weights [w_epsilon, w_mu] (pareto mode).",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Override control-effort budget (resilience mode).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print solver output.")
    parser.add_argument("--plot", action="store_true", help="Plot the closed-loop trajectory.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.N is not None:
        cfg["N"] = args.N
    if args.r is not None:
        cfg["r"] = args.r
    if args.w is not None:
        cfg["w"] = list(args.w)
    if args.epsilon is not None:
        cfg["epsilon"] = args.epsilon

    if args.mode == "effort":
        optimizer = build_effort_optimizer(cfg)
    elif args.mode == "pareto":
        optimizer = build_pareto_optimizer(cfg)
    else:  # resilience
        optimizer = build_resilience_optimizer(cfg)

    x0 = np.array(cfg["x0"], dtype=float)
    sol = optimizer.solve(x0, solver_name=args.solver, verbose=args.verbose)

    print("\n=== Results ===")
    print(f"Optimal epsilon : {sol['epsilon']}")
    if "mu" in sol:
        print(f"Optimal mu      : {sol['mu']}")
    print(f"Optimal alpha_1 :\n{sol['alpha_1']}")
    print(f"Optimal alpha_2 : {sol['alpha_2']}")

    if args.plot:
        rb = cfg.get("region_bounds", {})
        steps_sorted = sorted(int(k) for k in rb.keys())
        b1 = rb[str(steps_sorted[0])] if steps_sorted else cfg["safe_bounds"]
        b2 = rb[str(steps_sorted[1])] if len(steps_sorted) > 1 else cfg["safe_bounds"]
        plot_closed_loop_trajectories(
            optimizer, x0, b1=b1, b2=b2, safe_bounds=cfg["safe_bounds"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
