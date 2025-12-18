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

        Q = 15.0 * np.eye(nx)
        R = 1.0 * np.eye(nu)

        # Real-space input constraints: 40 <= Pavg <= 80
        M = np.array([[1.0], [-1.0]])
        m = np.array([80.0, -40.0])
        U_real = Polyhedron.from_Hrep(M, m)

        # Very loose state constraints (delta)
        v_max = 100000.0
        F = np.array([[1.0], [-1.0]])
        f = np.array([v_max, v_max])
        X_real = Polyhedron.from_Hrep(F, f)

        xs = self.xs.reshape(-1)
        us = self.us.reshape(-1)

        X = Polyhedron.from_Hrep(X_real.A, X_real.b - X_real.A @ xs)

        # Terminal weight only (no terminal set in Part 5.1)
        _, Qf, _ = dlqr(A, B, Q, R)

        # Estimator
        self.setup_estimator()

        # Variables/parameters (delta)
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")
        self.dx_ref_var = cp.Parameter((nx,), name="dx_ref")

        # Disturbance estimate parameter, used for input cancellation and constraints
        self.d_var = cp.Parameter((nu,), name="d")

        # Cost
        cost = 0
        for k in range(N):
            cost += cp.quad_form(self.dx_var[:, k] - self.dx_ref_var, Q)
            cost += cp.quad_form(self.du_var[:, k], R)
        cost += cp.quad_form(self.dx_var[:, -1] - self.dx_ref_var, Qf)

        # Constraints
        constraints = []
        constraints += [self.dx_var[:, 0] == self.dx0_var]

        # With cancellation u = us + du - d_hat, the predicted delta dynamics become nominal:
        # dx+ = A dx + B(du - d_hat) + B d_hat  = A dx + B du
        constraints += [self.dx_var[:, 1:] == A @ self.dx_var[:, :-1] + B @ self.du_var]

        constraints += [X.A @ self.dx_var[:, :-1] <= X.b.reshape(-1, 1)]

        # Applied input is u_apply = us + du - d_hat -> must satisfy 40..80
        d_col = cp.reshape(self.d_var, (nu, 1), order="F")
        u_apply = self.du_var + self.us.reshape(-1, 1) - d_col
        constraints += [U_real.A @ u_apply <= U_real.b.reshape(-1, 1)]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target=None, u_target=None):
        if self.u_prev is None:
            self.u_prev = self.us.copy()

        u_prev_scalar = float(np.atleast_1d(self.u_prev)[0])
        is_sat = (u_prev_scalar <= self.u_min + self.sat_eps) or (u_prev_scalar >= self.u_max - self.sat_eps)

        # Measurement in delta coords
        dx_meas = x0 - self.xs

        # Estimator uses the actually applied delta input: u_delta_applied = u_prev - us
        du_applied_prev = self.u_prev - self.us
        self.update_estimator(dx_meas, du_applied_prev, freeze_d=is_sat)

        # Use estimated state (delta) and estimated disturbance in MPC
        dx_hat = self.z_hat[: self.nx].reshape(-1)
        self.dx0_var.value = dx_hat
        self.d_var.value = self.d_estimate

        # MPC reference (delta)
        x_ref = self.xs if x_target is None else np.asarray(x_target).reshape(-1)
        self.dx_ref_var.value = (x_ref - self.xs)

        self.ocp.solve(solver=cp.PIQP)

        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u0 = self.us.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            self.u_prev = u0.copy()
            return u0, x_traj, u_traj

        dx_traj = self.dx_var.value
        du_traj = self.du_var.value

        # Real-space trajectories
        x_traj = dx_traj + self.xs.reshape(-1, 1)

        # Apply cancellation in the returned input trajectory
        d_hat = np.asarray(self.d_estimate).reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1) - d_hat
        u0 = u_traj[:, 0]

        self.u_prev = u0.copy()
        return u0, x_traj, u_traj

    def setup_estimator(self):
        nx, nu = self.nx, self.nu
        nd = nu

        self.u_prev = None
        self.u_min = 40.0
        self.u_max = 80.0
        self.sat_eps = 1e-3

        Bd = self.B

        self.A_hat = np.block([
            [self.A, Bd],
            [np.zeros((nd, nx)), np.eye(nd)]
        ])
        self.B_hat = np.vstack([self.B, np.zeros((nd, nu))])
        self.C_hat = np.hstack([np.eye(nx), np.zeros((nx, nd))])

        from scipy.signal import place_poles
        poles = np.array([0.83, 0.9])
        self.L = place_poles(self.A_hat.T, self.C_hat.T, poles).gain_matrix.T

        self.z_hat = np.zeros((nx + nd,))
        self.d_estimate = np.zeros((nd,))

    def update_estimator(self, x_meas: np.ndarray, u_applied: np.ndarray, freeze_d: bool = False) -> None:
        y = x_meas.reshape(-1)
        u = u_applied.reshape(-1)

        z_pred = self.A_hat @ self.z_hat + self.B_hat @ u
        innovation = y - (self.C_hat @ z_pred)
        z_new = z_pred + self.L @ innovation

        if freeze_d:
            z_new[self.nx:] = self.z_hat[self.nx:]

        self.z_hat = z_new
        self.d_estimate = self.z_hat[self.nx:]
