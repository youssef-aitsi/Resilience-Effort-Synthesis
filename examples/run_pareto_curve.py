"""
Example: sweep the disturbance bound r and plot the Pareto frontier.

For each r in a grid, the robust optimizer is solved and the optimal
``epsilon`` is recorded. The resulting (r, epsilon) curve is the Pareto
frontier between resilience (larger admissible r) and control effort
(larger epsilon).

Run with:
    python examples/run_pareto_curve.py
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ResilienceEffort import EffortOptimizer, plot_pareto_frontier


def main() -> None:
    # ------------------- System parameters -------------------
    N = 6   # Prediction horizon
    n = 2   # State dimension
    m = 2   # Input dimension
    q = 4   # Constraints per time step

    A = np.eye(2)
    B = np.eye(2)

    # ------------------- Optimizer ---------------------------
    optimizer = EffortOptimizer(N=N, n=n, m=m, q=q, A=A, B=B)

    G = np.array([[-1.0, 0.0],
                  [ 1.0, 0.0],
                  [ 0.0,-1.0],
                  [ 0.0, 1.0]])

    b1 = np.array([-0.3, 0.3, 0.6, 1.25])
    b2 = np.array([ 0.8, 1.5, 1.2, 1.75])
    b3 = np.array([-1.0, 1.7, 0.0, 2.0])
    H1 = np.array([-b1[0], b1[1], -b1[2], b1[3]])
    H2 = np.array([-b2[0], b2[1], -b2[2], b2[3]])
    H3 = np.array([-b3[0], b3[1], -b3[2], b3[3]])

    G_list = [G] * (N + 1)
    H_list = [H3] * (N + 1)
    H_list[2] = H1
    for k in range(4, len(H_list)):
        H_list[k] = H2

    optimizer.set_constraints(G_list, H_list)

    # ------------------- Sweep -------------------------------
    a, b, step = 0.0001, 0.06867, 0.0035
    r_values = np.linspace(a, b, int((b - a) / step) + 1)

    x0 = np.array([0.0, 0.2])
    r_arr, eps_arr = optimizer.pareto_sweep(x0, r_values, verbose=False)

    print(f"Solved {len(r_arr)} / {len(r_values)} points along the frontier.")
    for ri, ei in zip(r_arr, eps_arr):
        print(f"  r = {ri:.5f}   epsilon = {ei:.5f}")

    # ------------------- Plot --------------------------------
    plot_pareto_frontier(
        r_arr, eps_arr,
        selected=(0.0531, 0.55944),
        save_path="pareto_curve_double_integrator_closed_loop.png",
    )


if __name__ == "__main__":
    main()
