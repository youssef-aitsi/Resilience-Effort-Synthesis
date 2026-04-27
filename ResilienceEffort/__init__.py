"""
robust_mpc — Robust Model Predictive Control optimization with Pyomo.

Public API
----------
- ``RobustControlOptimizer``: the main optimizer class.
- ``plot_closed_loop_trajectories``: plotting helper for closed-loop runs.
"""

from .optimizer import EffortOptimizer, ParetoOptimizer, ResilienceOptimizer
from .plotting import plot_closed_loop_trajectories, plot_pareto_frontier

__all__ = [
    "EffortOptimizer",
    "ParetoOptimizer",
    "ResilienceOptimizer",
    "plot_closed_loop_trajectories",
    "plot_pareto_frontier",
]
__version__ = "0.1.0"
