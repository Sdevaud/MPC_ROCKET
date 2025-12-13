import numpy as np
from mpt4py import Polyhedron
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])   # [wy, beta, vx]
    u_ids: np.ndarray = np.array([1])         # delta2

    def _setup_controller(self) -> None:
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu

        Q = 10.0 * np.eye(nx)
        R = 1.0 * np.eye(nu)

        # ----- Real constraints (on x,u) -----
        # |delta2| <= 0.26
        M = np.array([[1.0], [-1.0]])
        m = np.array([0.26, 0.26])
        U_real = Polyhedron.from_Hrep(M, m)

        # |beta| <= 0.1745 (beta is reduced index 1)
        F = np.array([[0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.0]])
        f = np.array([0.1745, 0.1745])
        X_real = Polyhedron.from_Hrep(F, f)

        # ----- Shift constraints to DELTA space around (xs, us) -----
        xs = self.xs.reshape(-1)
        us = self.us.reshape(-1)

        X = Polyhedron.from_Hrep(X_real.A, X_real.b - X_real.A @ xs)
        U = Polyhedron.from_Hrep(U_real.A, U_real.b - U_real.A @ us)

        # Terminal controller / terminal set in DELTA space
        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        A_cl = A + B @ K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O_inf = self.max_invariant_set(A_cl, X.intersect(KU))

        # Variables in DELTA space
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")

        # Cost in DELTA space
        cost = 0
        for k in range(N):
            cost += cp.quad_form(self.dx_var[:, k], Q)
            cost += cp.quad_form(self.du_var[:, k], R)
        cost += cp.quad_form(self.dx_var[:, -1], Qf)

        constraints = []
        constraints += [self.dx_var[:, 0] == self.dx0_var]
        constraints += [self.dx_var[:, 1:] == A @ self.dx_var[:, :-1] + B @ self.du_var]
        constraints += [X.A @ self.dx_var[:, :-1] <= X.b.reshape(-1, 1)]
        constraints += [U.A @ self.du_var <= U.b.reshape(-1, 1)]
        constraints += [O_inf.A @ self.dx_var[:, -1] <= O_inf.b.reshape(-1, 1)]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target=None, u_target=None):
        # Work in DELTA space
        dx0 = x0 - self.xs
        self.dx0_var.value = dx0

        self.ocp.solve(solver=cp.PIQP)

        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u0 = self.us.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        dx_traj = self.dx_var.value
        du_traj = self.du_var.value

        # Convert back to REAL space
        x_traj = dx_traj + self.xs.reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1)
        u0 = u_traj[:, 0]
        return u0, x_traj, u_traj
