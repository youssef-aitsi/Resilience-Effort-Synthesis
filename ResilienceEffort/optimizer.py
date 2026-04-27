"""
Robust and Pareto control optimizers using Pyomo.

This module provides:

- :class:`_BaseOptimizer`  : shared infrastructure (constraint building,
  matrix construction, closed-loop simulation).
- :class:`EffortOptimizer` : minimizes the disturbance amplification
  ``epsilon`` for a fixed disturbance radius ``r``.
- :class:`ParetoOptimizer` : jointly trades off ``epsilon`` and the
  disturbance radius ``mu`` via a scalarized weighted objective
  ``w[0] * epsilon - w[1] * mu``.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    SolverFactory,
    minimize,
    maximize,
)


# ====================================================================== #
# Base class                                                              #
# ====================================================================== #
class _BaseOptimizer:
    """
    Shared scaffolding for the effort and Pareto optimizers.

    Subclasses must implement :meth:`_add_objective_and_scalar` to:

    1. Add their own decision variables (e.g., ``epsilon``, ``mu``).
    2. Set the Pyomo ``Objective``.
    3. Return the *scalar expression* that multiplies ``E`` on the right-hand
       side of the equality constraint ``P * A_b == scalar * E``.
    """

    def __init__(self, N: int, n: int, m: int, q: int, A: np.ndarray, B: np.ndarray):
        if A.shape != (n, n):
            raise ValueError(f"A must have shape ({n}, {n}), got {A.shape}")
        if B.shape != (n, m):
            raise ValueError(f"B must have shape ({n}, {m}), got {B.shape}")

        self.N = N
        self.n = n
        self.m = m
        self.q = q
        self.A = A
        self.B = B

        self.G_list: List[np.ndarray] = []
        self.H_list: List[np.ndarray] = []

        self.model: Optional[ConcreteModel] = None
        self.results = None

    # ------------------------------------------------------------------ #
    # Configuration                                                       #
    # ------------------------------------------------------------------ #
    def set_constraints(self, G_list: List[np.ndarray], H_list: List[np.ndarray]) -> None:
        """Set polytopic constraints ``G_list[k] @ x <= H_list[k]`` for each step k."""
        if len(G_list) != self.N + 1 or len(H_list) != self.N + 1:
            raise ValueError("Constraint lists must have length N+1")
        self.G_list = G_list
        self.H_list = H_list

    # ------------------------------------------------------------------ #
    # Internal: build problem matrices                                    #
    # ------------------------------------------------------------------ #
    def _build_problem_matrices(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct A_b, B_b, E, F (E and F contain Pyomo expressions)."""
        N, n, m, q = self.N, self.n, self.m, self.q

        # A_b: stacking +I and -I
        A_b = np.vstack((np.eye(n * N), -np.eye(n * N)))
        # A_u for input constraints
        A_u = np.vstack((np.eye(m), -np.eye(m)))
        # Ones vectors (for disturbance scaling)
        B_b = np.ones((2 * n * N, 1))
        B_u = np.ones((2 * m,))

        # Closed-loop A_bar = A + B * alpha_1 (kept symbolic via Pyomo Vars)
        A_bar = self.A + self.B @ self.model.alpha_1

        # ---------- E ----------
        E_blocks_x, E_blocks_u = [], []
        j = 0
        for k in range(1, N + 1):
            row_x, row_u = [], []
            for i in range(N):
                if i < k:
                    row_x.append(self.G_list[k] @ np.linalg.matrix_power(A_bar, k - 1 - i))
                else:
                    row_x.append(np.zeros((q, n)))

                if i < j:
                    row_u.append(
                        A_u @ self.model.alpha_1 @ np.linalg.matrix_power(A_bar, j - 1 - i)
                    )
                else:
                    row_u.append(np.zeros((2 * m, n)))

            E_blocks_x.append(np.hstack(row_x))
            E_blocks_u.append(np.hstack(row_u))
            j += 1

        E = np.vstack([np.vstack(E_blocks_x), np.vstack(E_blocks_u)])

        # ---------- F ----------
        F_x_parts, F_u_parts = [], []
        for k in range(1, N + 1):
            j = k - 1
            sum_A_k_B_alpha_2 = (
                sum(np.linalg.matrix_power(A_bar, k - 1 - i) for i in range(k))
                @ self.B
                @ self.model.alpha_2
            )
            sum_A_j_B_alpha_2 = (
                sum(
                    np.linalg.matrix_power(A_bar, j - 1 - i)
                    if j - 1 - i >= 0
                    else np.zeros_like(A_bar)
                    for i in range(k)
                )
                @ self.B
                @ self.model.alpha_2
            )

            F_k_x = (
                self.H_list[k]
                - self.G_list[k] @ np.linalg.matrix_power(A_bar, k) @ x
                - self.G_list[k] @ sum_A_k_B_alpha_2
            )
            S_k = (
                self.model.alpha_1 @ np.linalg.matrix_power(A_bar, j) @ x
                + self.model.alpha_1 @ sum_A_j_B_alpha_2
                + self.model.alpha_2
            )
            F_k_u = self.model.epsilon * B_u - A_u @ S_k

            F_x_parts.append(F_k_x)
            F_u_parts.append(F_k_u)

        F = np.hstack((np.hstack(F_x_parts), np.hstack(F_u_parts)))
        return A_b, B_b, E, F

    # ------------------------------------------------------------------ #
    # Hook to be implemented by subclasses                                #
    # ------------------------------------------------------------------ #
    def _add_objective_and_scalar(self):
        """Add objective + return the scalar expression multiplying E.

        Subclasses *must* (a) add ``self.model.epsilon`` (NonNegativeReals)
        and any extra variables, (b) attach an ``Objective`` named
        ``self.model.obj``, and (c) return the Pyomo expression that should
        multiply ``E`` in the equality constraint.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Solve                                                               #
    # ------------------------------------------------------------------ #
    def solve(
        self,
        x: np.ndarray,
        solver_name: str = "ipopt",
        verbose: bool = False,
    ):
        """Solve the optimization problem.

        Parameters
        ----------
        x : array-like
            Initial state of length ``n``.
        solver_name : str, optional
            Pyomo solver name. Default ``"ipopt"``. Use ``"ipopt"``
            if KNITRO is unavailable.
        verbose : bool, optional
            Print solver output if True.

        Returns
        -------
        dict
            Solution dict with keys ``epsilon``, ``alpha_1``, ``alpha_2``,
            ``P``, plus any subclass-specific keys (e.g. ``mu``).
        """
        x = np.asarray(x, dtype=float).flatten()
        if len(x) != self.n:
            raise ValueError(f"Initial state must have dimension {self.n}")
        if not self.G_list or not self.H_list:
            raise ValueError("Constraints not set. Call set_constraints() first.")

        self.model = ConcreteModel()

        qN = self.q * self.N
        nN = self.n * self.N
        two_nN = 2 * self.n * self.N
        two_mN = 2 * self.m * self.N

        # Common variables
        self.model.epsilon = Var(within=NonNegativeReals)
        self.model.P = Var(range(qN + two_mN), range(two_nN), within=NonNegativeReals)
        self.model.alpha_1 = Var(range(self.m), range(self.n), initialize=0.1)
        self.model.alpha_2 = Var(range(self.m), initialize=0.1)

        # Subclass-specific: extra vars + objective + scalar that multiplies E
        scalar_expr = self._add_objective_and_scalar()

        # Build problem matrices (depend on Pyomo Vars; do this AFTER var declarations)
        A_b, B_b, E, F = self._build_problem_matrices(x)

        # Equality constraint: P * A_b == scalar * E
        def pa_eq_e_rule(model, i, j):
            lhs = sum(model.P[i, k] * A_b[k, j] for k in range(two_nN))
            return lhs == scalar_expr * E[i, j]

        self.model.pa_eq_e = Constraint(
            range(qN + two_mN), range(nN), rule=pa_eq_e_rule
        )

        # Inequality constraint: P * B_b <= F
        def pb_le_f_rule(model, i):
            lhs = sum(model.P[i, j] * B_b[j, 0] for j in range(two_nN))
            return lhs <= F[i]

        self.model.pb_le_f = Constraint(range(qN + two_mN), rule=pb_le_f_rule)

        # Solve
        solver = SolverFactory(solver_name)
        if not verbose:
            try:
                solver.options["print_level"] = 1
            except Exception:
                pass
        self.results = solver.solve(self.model, tee=verbose)

        return self._extract_solution()

    # ------------------------------------------------------------------ #
    # Solution extraction                                                 #
    # ------------------------------------------------------------------ #
    def _extract_common(self) -> dict:
        qN = self.q * self.N
        two_nN = 2 * self.n * self.N
        two_mN = 2 * self.m * self.N

        alpha_1 = np.array(
            [[self.model.alpha_1[i, j]() for j in range(self.n)] for i in range(self.m)]
        )
        alpha_2 = np.array([self.model.alpha_2[i]() for i in range(self.m)])
        P = np.zeros((qN + two_mN, two_nN))
        for i in range(qN + two_mN):
            for j in range(two_nN):
                P[i, j] = self.model.P[i, j]()

        return {
            "epsilon": self.model.epsilon(),
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "P": P,
        }

    def _extract_solution(self) -> dict:
        """Override in subclasses to add solver-specific keys."""
        return self._extract_common()

    # ------------------------------------------------------------------ #
    # Closed-loop simulation                                              #
    # ------------------------------------------------------------------ #
    def _disturbance_radius(self) -> float:
        """The constant added to each step in the closed-loop simulation."""
        raise NotImplementedError

    def simulate_closed_loop(
        self, x0: np.ndarray, nominal: bool = False
    ) -> np.ndarray:
        """Simulate the closed-loop trajectory using ``u = alpha_1*x + alpha_2``."""
        if self.model is None:
            raise RuntimeError("Solve the optimization problem before simulating.")

        alpha_1 = np.array(
            [[self.model.alpha_1[i, j]() for j in range(self.n)] for i in range(self.m)]
        )
        alpha_2 = np.array([self.model.alpha_2[i]() for i in range(self.m)])
        r = 0.0 if nominal else self._disturbance_radius()

        x_series = [np.asarray(x0, dtype=float).flatten()]
        for _ in range(self.N):
            x_k = x_series[-1]
            x_next = self.A @ x_k + self.B @ (alpha_1 @ x_k + alpha_2) + r
            x_series.append(x_next)
        return np.array(x_series)


# ====================================================================== #
# Effort optimizer: fixed r, minimize epsilon                             #
# ====================================================================== #
class EffortOptimizer(_BaseOptimizer):
    """Minimize ``epsilon`` for a fixed disturbance radius ``r``.

    Workflow::

        opt = EffortOptimizer(N, n, m, q, A, B)
        opt.set_constraints(G_list, H_list)
        opt.set_disturbance_bound(r=0.05)
        sol = opt.solve(x0)
        # sol = {"epsilon": ..., "alpha_1": ..., "alpha_2": ..., "P": ...}
    """

    def __init__(self, N, n, m, q, A, B):
        super().__init__(N, n, m, q, A, B)
        self.r: float = 0.0

    def set_disturbance_bound(self, r: float) -> None:
        """Set the (non-negative) disturbance bound radius."""
        if r < 0:
            raise ValueError("Disturbance bound must be non-negative")
        self.r = float(r)

    def _add_objective_and_scalar(self):
        # Objective: minimize epsilon
        self.model.obj = Objective(expr=self.model.epsilon, sense=minimize)
        # Scalar multiplying E is the fixed parameter r
        return self.r

    def _disturbance_radius(self) -> float:
        return self.r

    # ------------------------------------------------------------------ #
    # Pareto frontier sweep                                               #
    # ------------------------------------------------------------------ #
    def pareto_sweep(
        self,
        x0: np.ndarray,
        r_values: np.ndarray,
        solver_name: str = "ipopt",
        verbose: bool = False,
        skip_infeasible: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the problem for a range of disturbance bounds.

        For each ``r`` in ``r_values``, set the disturbance bound, solve
        the optimization problem from ``x0``, and record the resulting
        optimal ``epsilon``. The output traces the Pareto frontier between
        resilience (``r``) and control effort (``epsilon``).

        Parameters
        ----------
        x0 : array-like
            Initial state.
        r_values : array-like
            Sequence of disturbance bounds to sweep.
        solver_name, verbose : forwarded to :meth:`solve`.
        skip_infeasible : bool, optional
            If True (default), entries whose solver fails are dropped from
            the returned arrays. If False, they are kept as ``nan``.

        Returns
        -------
        r_array, epsilon_array : tuple of np.ndarray
            The (filtered) ``r`` values and matching optimal ``epsilon``
            values.
        """
        r_values = np.asarray(r_values, dtype=float)
        epsilons = []
        for r in r_values:
            self.set_disturbance_bound(r=float(r))
            try:
                sol = self.solve(x0, solver_name=solver_name, verbose=verbose)
                epsilons.append(sol["epsilon"])
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[pareto_sweep] r={r:.5f} failed: {exc}")
                epsilons.append(np.nan)

        epsilon_array = np.array(epsilons, dtype=float)
        if skip_infeasible:
            mask = ~np.isnan(epsilon_array)
            return r_values[mask], epsilon_array[mask]
        return r_values, epsilon_array


# ====================================================================== #
# Pareto optimizer: minimize w0*epsilon - w1*mu                           #
# ====================================================================== #
class ParetoOptimizer(_BaseOptimizer):
    """Pareto trade-off between ``epsilon`` and the disturbance radius ``mu``.

    The objective is::

        minimize  w[0] * epsilon  -  w[1] * mu

    where both ``epsilon`` and ``mu`` are decision variables. Larger
    ``w[0]`` favors smaller amplification; larger ``w[1]`` favors a larger
    admissible disturbance.

    Workflow::

        opt = ParetoOptimizer(N, n, m, q, A, B)
        opt.set_constraints(G_list, H_list)
        opt.set_weights(w=[0.02, 0.1])
        sol = opt.solve(x0)
        # sol = {"epsilon": ..., "mu": ..., "alpha_1": ..., "alpha_2": ..., "P": ...}
    """

    def __init__(self, N, n, m, q, A, B):
        super().__init__(N, n, m, q, A, B)
        self.w: Sequence[float] = (1.0, 1.0)

    def set_weights(self, w: Sequence[float]) -> None:
        """Set the two-element weight vector ``[w_epsilon, w_mu]``."""
        if len(w) != 2:
            raise ValueError("w must have length 2: [w_epsilon, w_mu]")
        if w[0] <= 0 or w[1] <= 0:
            raise ValueError("Weights must be positive")
        self.w = (float(w[0]), float(w[1]))

    # Backward-compatible alias for the original name
    def set_disturbance_weight(self, w: Sequence[float]) -> None:
        """Alias for :meth:`set_weights` (kept for backward compatibility)."""
        self.set_weights(w)

    def _add_objective_and_scalar(self):
        # Extra variable: mu (also non-negative)
        self.model.mu = Var(within=NonNegativeReals)
        # Objective: w0 * epsilon - w1 * mu (minimize)
        self.model.obj = Objective(
            expr=self.w[0] * self.model.epsilon - self.w[1] * self.model.mu,
            sense=minimize,
        )
        # Scalar multiplying E is mu (a decision variable here)
        return self.model.mu

    def _extract_solution(self) -> dict:
        sol = self._extract_common()
        sol["mu"] = self.model.mu()
        return sol

    def _disturbance_radius(self) -> float:
        return self.model.mu()


# ====================================================================== #
# Resilience optimizer: fixed epsilon, maximize mu                        #
# ====================================================================== #
class ResilienceOptimizer(_BaseOptimizer):
    """Maximize the disturbance radius ``mu`` for a fixed control effort ``epsilon``.

    This is the dual of :class:`EffortOptimizer`: instead of fixing
    the disturbance radius and minimizing control effort, you fix the
    available control effort budget and find the largest disturbance the
    system can tolerate.

    Workflow::

        opt = ResilienceOptimizer(N, n, m, q, A, B)
        opt.set_constraints(G_list, H_list)
        opt.set_input_budget(epsilon=0.93)
        sol = opt.solve(x0)
        # sol = {"epsilon": ..., "mu": ..., "alpha_1": ..., "alpha_2": ..., "P": ...}

    Note that ``sol["epsilon"]`` is just the fixed input value back; the
    quantity actually solved for is ``sol["mu"]``.
    """

    def __init__(self, N, n, m, q, A, B):
        super().__init__(N, n, m, q, A, B)
        self.e: float = 1.0

    def set_input_budget(self, epsilon: float) -> None:
        """Set the fixed control-effort budget (``epsilon``).

        Larger values give the optimizer more "control room" and therefore
        admit a larger maximum disturbance ``mu``.
        """
        if epsilon <= 0:
            raise ValueError("epsilon (control-effort budget) must be positive")
        self.e = float(epsilon)

    # Backward-compatible alias matching the original API name
    def set_input(self, e: float) -> None:
        """Alias for :meth:`set_input_budget`."""
        self.set_input_budget(e)

    def _add_objective_and_scalar(self):
        # Add mu as a (free, non-negative) decision variable.
        self.model.mu = Var(within=NonNegativeReals)

        # Fix epsilon to the user-supplied input budget. We declare it as a
        # Var (the base _build_problem_matrices code references
        # self.model.epsilon as if it were one) and then fix its value, so
        # the solver treats it as a parameter.
        self.model.epsilon.fix(self.e)

        # Objective: maximize mu
        self.model.obj = Objective(expr=self.model.mu, sense=maximize)

        # Scalar multiplying E is mu
        return self.model.mu

    def _extract_solution(self) -> dict:
        sol = self._extract_common()
        sol["epsilon"] = self.e          # echo the fixed input budget
        sol["mu"] = self.model.mu()
        return sol

    def _disturbance_radius(self) -> float:
        return self.model.mu()
