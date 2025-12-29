import numpy as np
from mpt4py import Polyhedron
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])  # [vz]
    u_ids: np.ndarray = np.array([2])  # Pavg

    def _setup_controller(self) -> None:
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu

        Q = 50.0 * np.eye(nx)
        R = 1.0 * np.eye(nu)

        #Real space constraints
        #40 <= Pavg <= 80
        M = np.array([[1.0], [-1.0]])
        m = np.array([80.0, -40.0])
        U_real = Polyhedron.from_Hrep(M, m)

        #Need a finite state set for terminal invariant computation so we set very loose constraints
        v_max = 100000.
        F = np.array([[1.0], [-1.0]])
        f = np.array([v_max, v_max])
        X_real = Polyhedron.from_Hrep(F, f)

        #Delta space constraints
        xs = self.xs.reshape(-1)
        us = self.us.reshape(-1)

        X = Polyhedron.from_Hrep(X_real.A, X_real.b - X_real.A @ xs)
        U = Polyhedron.from_Hrep(U_real.A, U_real.b - U_real.A @ us)

        #Terminal controller and terminal set in delta space
        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        A_cl = A + B @ K
        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        O_inf = self.max_invariant_set(A_cl, X.intersect(KU))

        #Variables in delta space
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")
        self.dx_ref_var = cp.Parameter((nx,), name="dx_ref")

        #Cost function in delta space
        cost = 0
        for k in range(N):
            cost += cp.quad_form(self.dx_var[:, k]- self.dx_ref_var, Q)
            cost += cp.quad_form(self.du_var[:, k], R)
        cost += cp.quad_form(self.dx_var[:, -1]- self.dx_ref_var, Qf)

        constraints = []
        constraints += [self.dx_var[:, 0] == self.dx0_var]
        constraints += [self.dx_var[:, 1:] == A @ self.dx_var[:, :-1] + B @ self.du_var]
        constraints += [X.A @ self.dx_var[:, :-1] <= X.b.reshape(-1, 1)]
        constraints += [U.A @ self.du_var <= U.b.reshape(-1, 1)]
        constraints += [O_inf.A @ self.dx_var[:, -1] <= O_inf.b.reshape(-1, 1)]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target=None, u_target=None):
        #delta space solving
        dx0 = x0 - self.xs
        self.dx0_var.value = dx0

        if x_target is None:
            x_ref = self.xs
        else:
            x_target = np.asarray(x_target).reshape(-1)
            x_ref = x_target

        dx_ref = x_ref - self.xs
        self.dx_ref_var.value = dx_ref

        self.ocp.solve(solver=cp.PIQP)

        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u0 = self.us.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        dx_traj = self.dx_var.value
        du_traj = self.du_var.value

        #Convert back to real space
        x_traj = dx_traj + self.xs.reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1)   # <-- ensures Pavg is 40..80 in real space
        u0 = u_traj[:, 0]
        return u0, x_traj, u_traj
