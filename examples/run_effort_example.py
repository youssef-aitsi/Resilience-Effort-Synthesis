"""
Example: 2D robust MPC with two target regions and a safe region.

This script mirrors the original demonstration: a 2-state, 2-input system
(the identity dynamics) is steered through two intermediate target regions
while remaining inside a safe region under bounded disturbance.

Run with:
    python examples/run_effort_example.py
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ResilienceEffort import EffortOptimizer, plot_closed_loop_trajectories


def main() -> None:
    # ------------------- System parameters -------------------
    N = 6   # Prediction horizon
    n = 2   # State dimension
    m = 2   # Input dimension
    q = 4   # Constraints per time step

    A = np.array([[1.0, 0.0],
                  [0.0, 1.0]])  # System matrix
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0]])  # Control matrix

    # ------------------- Optimizer ---------------------------
    optimizer = EffortOptimizer(N=N, n=n, m=m, q=q, A=A, B=B)

    # Constraint matrix (x1, x2 box constraints, repeated each step)
    G = np.array([[-1, 0],
                  [ 1, 0],
                  [ 0,-1],
                  [ 0, 1]])

    # Region bounds: [xmin, xmax, ymin, ymax]
    b1 = np.array([-0.3, 0.3, 0.6, 1.25])     # Intermediate region 1
    b2 = np.array([ 0.8, 1.5, 1.2, 1.75])     # Intermediate region 2
    b3 = np.array([-1.0, 1.7, 0.0, 2.0])      # Safe region (outer box)

    # Convert [xmin, xmax, ymin, ymax] to G @ x <= H form:  H = [-xmin, xmax, -ymin, ymax]
    H1 = np.array([-b1[0], b1[1], -b1[2], b1[3]])
    H2 = np.array([-b2[0], b2[1], -b2[2], b2[3]])
    H3 = np.array([-b3[0], b3[1], -b3[2], b3[3]])

    # Default to safe region for every time step, then override specific steps
    G_list = [G] * (N + 1)
    H_list = [H3] * (N + 1)
    H_list[2] = H1                        # Reach region 1 at step 2
    for k in range(4, len(H_list)):       # Stay in region 2 from step 4 onwards
        H_list[k] = H2

    optimizer.set_constraints(G_list, H_list)
    optimizer.set_disturbance_bound(r=0.068678650071773)

    # ------------------- Solve -------------------------------
    x0 = np.array([0.0, 0.2])
    sol = optimizer.solve(x0, verbose=True)

    print("\n=== Results ===")
    print(f"Optimal epsilon : {sol['epsilon']}")
    print(f"Optimal alpha_1 :\n{sol['alpha_1']}")
    print(f"Optimal alpha_2 : {sol['alpha_2']}")

    # ------------------- Plot --------------------------------
    plot_closed_loop_trajectories(optimizer, x0, b1=b1, b2=b2, safe_bounds=b3)


if __name__ == "__main__":
    main()
