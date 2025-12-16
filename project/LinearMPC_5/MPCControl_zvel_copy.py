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

        # --- Real space input constraints: 40 <= Pavg <= 80
        M = np.array([[1.0], [-1.0]])
        m = np.array([80.0, -40.0])
        U_real = Polyhedron.from_Hrep(M, m)

        # --- Real space state constraints (loose)
        v_max = 100000.0
        F = np.array([[1.0], [-1.0]])
        f = np.array([v_max, v_max])
        X_real = Polyhedron.from_Hrep(F, f)

        # --- Delta space state constraints
        xs = self.xs.reshape(-1)
        us = self.us.reshape(-1)
        X = Polyhedron.from_Hrep(X_real.A, X_real.b - X_real.A @ xs)

        # Terminal weight
        _, Qf, _ = dlqr(A, B, Q, R)

        # Estimator
        self.setup_estimator()

        # --- Decision variables / parameters (delta space)
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")
        self.dx_ref_var = cp.Parameter((nx,), name="dx_ref")

        # disturbance estimate (constant over horizon), same dimension as input
        self.d_var = cp.Parameter((nu,), name="d")

        # --- Cost
        cost = 0
        for k in range(N):
            cost += cp.quad_form(self.dx_var[:, k] - self.dx_ref_var, Q)
            cost += cp.quad_form(self.du_var[:, k], R)
        cost += cp.quad_form(self.dx_var[:, -1] - self.dx_ref_var, Qf)

        # --- Constraints
        constraints = []
        constraints += [self.dx_var[:, 0] == self.dx0_var]

        # Dynamics with additive "input-channel disturbance":
        # dx+ = A dx + B du + B d
        constraints += [
            self.dx_var[:, 1:] == A @ self.dx_var[:, :-1]
            + B @ self.du_var
            + B @ cp.reshape(self.d_var, (nu, 1), order="F")
        ]

        # State constraints (delta)
        constraints += [X.A @ self.dx_var[:, :-1] <= X.b.reshape(-1, 1)]

        # IMPORTANT (consistent choice):
        # We do NOT cancel d_hat in the applied input.
        # So the applied input is simply u_apply = us + du, and this is what must satisfy 40..80.
        u_apply = self.du_var + self.us.reshape(-1, 1)
        constraints += [U_real.A @ u_apply <= U_real.b.reshape(-1, 1)]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target=None, u_target=None):
        # Initialize previous applied input (real)
        if self.u_prev is None:
            self.u_prev = self.us.copy()

        # Saturation flag based on previously APPLIED input
        u_prev_scalar = float(np.atleast_1d(self.u_prev)[0])
        is_sat = (u_prev_scalar <= self.u_min + self.sat_eps) or (u_prev_scalar >= self.u_max - self.sat_eps)

        # Estimator update in DELTA coordinates, but freeze d_hat if saturated (anti-windup)
        dx_meas = x0 - self.xs
        du_prev = self.u_prev - self.us
        self.update_estimator(dx_meas, du_prev, freeze_d=is_sat)
        print("dhat = ", self.d_estimate)
        # Use updated estimate in MPC prediction
        self.d_var.value = self.d_estimate

        # MPC initial condition + reference (delta)
        self.dx0_var.value = dx_meas
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

        # Convert back to real space
        x_traj = dx_traj + self.xs.reshape(-1, 1)

        # IMPORTANT (consistent choice):
        # Do NOT subtract d_hat here. Applied input is u = us + du.
        u_traj = du_traj + self.us.reshape(-1, 1)
        u0 = u_traj[:, 0]

        # Store APPLIED input for next estimator update call (k+1)
        self.u_prev = u0.copy()

        
        return u0, x_traj, u_traj

    def setup_estimator(self):
        nx, nu = self.nx, self.nu
        nd = nu

        self.u_prev = None
        self.u_min = 40.0
        self.u_max = 80.0
        self.sat_eps = 1e-3

        Bd = self.B  # disturbance enters like input

        # z = [x; d], with d^+ = d
        self.A_hat = np.block([
            [self.A, Bd],
            [np.zeros((nd, nx)), np.eye(nd)]
        ])
        self.B_hat = np.vstack([self.B, np.zeros((nd, nu))])

        # y = x (measuring the subsystem state)
        self.C_hat = np.hstack([np.eye(nx), np.zeros((nx, nd))])

        from scipy.signal import place_poles
        # Recommended: slow-ish poles to avoid noisy d_hat. (For nx+nd=2 these are two values.)
        poles = np.array([0.9, 0.95]) if (nx + nd) == 2 else np.linspace(0.85, 0.95, nx + nd)
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
            # freeze ONLY the disturbance states (anti-windup during saturation)
            z_new[self.nx:] = self.z_hat[self.nx:]

        self.z_hat = z_new
        self.d_estimate = self.z_hat[self.nx:]
