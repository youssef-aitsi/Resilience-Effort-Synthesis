"""
Example: 2D resilience optimizer with two target regions and a safe region.

This is the dual of the robust example: instead of fixing the disturbance
radius and minimizing the control effort ``epsilon``, here ``epsilon`` is
fixed (the available "control budget") and we maximize the largest
disturbance radius ``mu`` the closed-loop system can absorb.

Run with:
    python examples/run_resilience_example.py
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ResilienceEffort import ResilienceOptimizer, plot_closed_loop_trajectories


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
    optimizer = ResilienceOptimizer(N=N, n=n, m=m, q=q, A=A, B=B)

    G = np.array([[-1.0, 0.0],
                  [ 1.0, 0.0],
                  [ 0.0,-1.0],
                  [ 0.0, 1.0]])

    # Region bounds: [xmin, xmax, ymin, ymax]
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

    # Fix the control-effort budget; the solver will find the largest mu
    # (disturbance radius) the closed-loop system can absorb under that budget.
    optimizer.set_input_budget(epsilon=0.93)

    # ------------------- Solve -------------------------------
    x0 = np.array([0.0, 0.2])
    sol = optimizer.solve(x0, solver_name="ipopt", verbose=True)

    print("\n=== Results ===")
    print(f"Fixed epsilon (input)  : {sol['epsilon']}")
    print(f"Optimal mu (resilience): {sol['mu']}")
    print(f"Optimal alpha_1        :\n{sol['alpha_1']}")
    print(f"Optimal alpha_2        : {sol['alpha_2']}")

    # ------------------- Plot --------------------------------
    plot_closed_loop_trajectories(optimizer, x0, b1=b1, b2=b2, safe_bounds=b3)


if __name__ == "__main__":
    main()
