import numpy as np
from mpt4py import Polyhedron
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])  # vz
    u_ids: np.ndarray = np.array([2])  # Pavg

    d_estimate: float
    d_gain: float

    def _setup_controller(self) -> None:
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu  # nx=1, nu=1

        Q = 15.0 * np.eye(nx)
        R = 1.0 * np.eye(nu)

        # Real input constraints: 40 <= Pavg <= 80
        M = np.array([[1.0], [-1.0]])
        m = np.array([80.0, -40.0])
        U_real = Polyhedron.from_Hrep(M, m)

        us = self.us.reshape(-1)
        U = Polyhedron.from_Hrep(U_real.A, U_real.b - U_real.A @ us)

        # Terminal cost
        _, Qf, _ = dlqr(A, B, Q, R)

        # Delta MPC vars
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")
        self.dx_ref_var = cp.Parameter((nx,), name="dx_ref")

        # Additive constant disturbance in dynamics (Part 5.1)
        # dx_{k+1} = A dx_k + B du_k + d_hat
        self.dhat_var = cp.Parameter((1,), name="dhat")

        cost = 0
        constraints = [self.dx_var[:, 0] == self.dx0_var]

        for k in range(N):
            cost += cp.quad_form(self.dx_var[:, k] - self.dx_ref_var, Q)
            cost += cp.quad_form(self.du_var[:, k], R)

            constraints += [self.dx_var[:, k + 1] == A @ self.dx_var[:, k] + B @ self.du_var[:, k]]

        cost += cp.quad_form(self.dx_var[:, N] - self.dx_ref_var, Qf)

        constraints += [U.A @ self.du_var <= U.b.reshape(-1, 1)]
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # Estimator init
        self.d_estimate = 0.0
        self.d_gain = 0.05    # 0.05..0.2 typical
        self._x_prev = None
        self._u_prev = float(us[0])
        self._d_clip = 5   # wide in state-units

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        x0 = float(np.asarray(x0).reshape(-1)[0])

        # vz reference (Deliverable uses v_ref=[0,0,0] => 0)
        if x_target is None:
            x_ref = 0.0
        else:
            x_ref = float(np.asarray(x_target).reshape(-1)[0])

        # --- disturbance estimator update (SIGN FIX HERE) ---
        # model: x_k ≈ A x_{k-1} + B u_{k-1} + d
        if self._x_prev is None:
            self._x_prev = x0
        else:
            A = float(np.asarray(self.A).reshape(1)[0])
            B = float(np.asarray(self.B).reshape(1)[0])

            x_pred = A * self._x_prev + B * self._u_prev + self.d_estimate

            # ✅ SIGN FIX: innovation = predicted - measured
            err = x_pred - x0

            self.d_estimate = float(self.d_estimate + self.d_gain * err)
            self.d_estimate = float(np.clip(self.d_estimate, -self._d_clip, self._d_clip))

            self._x_prev = x0

        # --- MPC params (delta around trim) ---
        xs = float(self.xs.reshape(-1)[0])
        self.dx0_var.value = np.array([x0 - xs], dtype=float)
        self.dx_ref_var.value = np.array([x_ref - xs - self.d_estimate], dtype=float)
        self.dhat_var.value = np.array([self.d_estimate], dtype=float)

        self.ocp.solve(solver=cp.PIQP)

        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u0 = self.us.copy()
            x_traj = np.tile(np.array([[x0]]), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            self._u_prev = float(u0.reshape(-1)[0])
            return u0, x_traj, u_traj

        du0 = float(self.du_var.value[:, 0])
        u0 = du0 + float(self.us.reshape(-1)[0])

        # store applied REAL input for next estimator step
        self._u_prev = float(u0)

        x_traj = np.asarray(self.dx_var.value, dtype=float) + xs
        u_traj = np.asarray(self.du_var.value, dtype=float) + float(self.us.reshape(-1)[0])

        return np.array([u0], dtype=float), x_traj, u_traj
