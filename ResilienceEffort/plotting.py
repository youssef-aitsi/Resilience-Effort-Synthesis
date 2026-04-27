"""
Plotting utilities for robust control trajectories.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


def _add_rectangle(
    ax: plt.Axes,
    bounds: Sequence[float],
    edgecolor: str = "black",
    facecolor: str = "none",
    alpha: float = 0.5,
    linewidth: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Add a rectangular polytope to the given axes.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to draw on.
    bounds : sequence of 4 floats
        ``[xmin, xmax, ymin, ymax]`` defining the rectangle.
    edgecolor, facecolor, alpha, linewidth, label
        Standard matplotlib styling parameters.
    """
    xmin, xmax, ymin, ymax = bounds
    vertices = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    )
    patch = Polygon(
        vertices,
        closed=True,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(patch)


def plot_closed_loop_trajectories(
    optimizer,
    x0: np.ndarray,
    b1: Sequence[float],
    b2: Sequence[float],
    safe_bounds: Sequence[float] = (-1, 1.7, 0, 2),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    """
    Plot closed-loop trajectories (effort and nominal) along with constraint
    polytopes.

    Parameters
    ----------
    optimizer : Optimizer
        A solved optimizer instance.
    x0 : np.ndarray
        Initial state.
    b1, b2 : sequence of 4 floats
        Bounds ``[xmin, xmax, ymin, ymax]`` for the two intermediate target
        regions.
    safe_bounds : sequence of 4 floats, optional
        Bounds of the safe region (drawn as an outline). Default is
        ``(-1, 1.7, 0, 2)``.
    ax : plt.Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.
    show : bool, optional
        Whether to call ``plt.show()`` at the end. Default True.

    Returns
    -------
    plt.Axes
        The axes object that was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    # Polytopes
    _add_rectangle(ax, b1, edgecolor="red", facecolor="lightpink", alpha=0.5)
    _add_rectangle(ax, b2, edgecolor="blue", facecolor="lightblue", alpha=0.5)
    _add_rectangle(
        ax,
        safe_bounds,
        edgecolor="black",
        facecolor="none",
        alpha=1.0,
        linewidth=2.0,
        label="Safe Region",
    )

    # Trajectories
    trajectory = optimizer.simulate_closed_loop(x0, nominal=False)
    trajectory_nominal = optimizer.simulate_closed_loop(x0, nominal=True)

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="blue",
        linestyle="-",
        marker="o",
        markersize=5,
    )
    ax.plot(
        trajectory_nominal[:, 0],
        trajectory_nominal[:, 1],
        color="orange",
        linestyle="-",
        marker="o",
        markersize=5,
    )

    # Highlight key points
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1],
        color="white", s=50, edgecolor="black", zorder=2, label="Start",
    )
    if len(trajectory) > 2:
        ax.scatter(
            trajectory[2, 0], trajectory[2, 1],
            color="red", s=30, edgecolor="black", zorder=2, label="time step 2",
        )
        ax.scatter(
            trajectory_nominal[2, 0], trajectory_nominal[2, 1],
            color="red", s=30, edgecolor="black", zorder=2,
        )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1],
        color="black", s=30, edgecolor="black", zorder=5, label="End",
    )

    # Trajectory legend (separate from start/end legend)
    traj_legend_elements = [
        Line2D(
            [0], [0], color="blue", linestyle="-", marker="o", markersize=5,
            label=r"Trajectory with $\varepsilon > \varepsilon_{\max}$",
        ),
        Line2D(
            [0], [0], color="orange", linestyle="-", marker="o", markersize=5,
            label="Nominal Trajectory",
        ),
    ]

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    main_legend = ax.legend(
        unique_handles, unique_labels, loc="upper left", bbox_to_anchor=(0.05, 0.95)
    )
    ax.add_artist(main_legend)
    ax.legend(handles=traj_legend_elements, loc="lower right", bbox_to_anchor=(0.95, 0.05))

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax


def plot_pareto_frontier(
    r_values: np.ndarray,
    epsilon_values: np.ndarray,
    ax: Optional[plt.Axes] = None,
    selected: Optional[Sequence[float]] = None,
    y_max: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the Pareto frontier between resilience and control effort.

    Below the frontier is shaded red (infeasible); above is shaded green
    (feasible). The first point on the frontier is highlighted in red.

    Parameters
    ----------
    r_values, epsilon_values : array-like
        Matching arrays from :meth:`EffortOptimizer.pareto_sweep`.
    ax : plt.Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.
    selected : (r, epsilon) tuple, optional
        If provided, draw a single highlighted "selected solution" marker.
    y_max : float, optional
        Upper limit of the y-axis. Defaults to ``1.25 * max(epsilon)``.
    save_path : str, optional
        If provided, save the figure to this path with ``dpi=300``.
    show : bool, optional
        Whether to call ``plt.show()`` at the end.

    Returns
    -------
    plt.Axes
        The axes object that was drawn on.
    """
    r = np.asarray(r_values, dtype=float)
    e = np.asarray(epsilon_values, dtype=float)

    if r.shape != e.shape or r.ndim != 1:
        raise ValueError("r_values and epsilon_values must be 1-D arrays of equal length")
    if len(r) < 2:
        raise ValueError("Need at least 2 points to plot a frontier")

    order = np.argsort(r)
    r = r[order]
    e = e[order]

    if y_max is None:
        y_max = float(np.nanmax(e)) * 1.25

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Infeasible region (below the frontier)
    ax.fill_between(
        r, 0, e, color=(1.0, 0.4, 0.4), alpha=0.3, label="Infeasible Region"
    )
    # Feasible region (above the frontier, up to y_max)
    ax.fill_between(
        r, e, y_max, color="green", alpha=0.25, label="Feasible Region"
    )

    # Pareto frontier line
    ax.plot(
        r, e,
        color="blue", linewidth=2, marker="o", markersize=6,
        label="Pareto Frontier", zorder=3,
    )

    # Highlight first point
    ax.scatter(
        [r[0]], [e[0]],
        color="red", edgecolor="darkred", s=50, linewidth=2, zorder=5,
    )

    # Optional selected solution
    if selected is not None:
        sel_r, sel_e = float(selected[0]), float(selected[1])
        ax.scatter(
            [sel_r], [sel_e],
            color="black", edgecolor="black",
            s=150, marker=".", linewidth=2, zorder=6,
            label="Selected Solution",
        )

    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(max(0.0, np.nanmin(e) - 0.05), y_max)
    ax.set_xlabel(r"Resilience $\mu$", fontsize=14)
    ax.set_ylabel(r"Control Effort $\varepsilon$", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.tick_params(axis="x", labelrotation=45)

    if save_path is not None:
        plt.gcf().savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return ax
