import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""
    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition (reduce full A,B to subsystem indices)
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    @staticmethod
    def max_invariant_set(A_cl, X: Polyhedron, max_iter=30) -> Polyhedron:
        O = X
        for _ in range(max_iter):
            Oprev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.hstack((f, f)))
            O.minHrep(True)
            _ = O.Vrep
            if O == Oprev:
                break
        return O

    def _setup_controller(self) -> None:
        """
        Generic nominal MPC in delta coordinates:
            dx+ = A dx + B du
        Cost:
            sum dx'Qdx + du'Rdu + terminal dx'Qf dx
        This is used for the nominal controllers (x,y,roll) and as a safe base.
        Subclasses (like robust tube MPC) can override this completely.
        """
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu

        # Default weights (reasonable generic values)
        Q = np.eye(nx)
        R = 0.1 * np.eye(nu)

        # Terminal cost from LQR
        K_lqr, Qf, _ = dlqr(A, B, Q, R)

        # Decision variables (delta form)
        x = cp.Variable((nx, N + 1), name="x")
        u = cp.Variable((nu, N), name="u")

        # Parameter: initial delta state
        self.dx0_var = cp.Parameter((nx,), name="dx0")

        cost = 0
        constr = [x[:, 0] == self.dx0_var]

        for k in range(N):
            cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
            constr += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]

        cost += cp.quad_form(x[:, N], Qf)

        # Store to use in get_u()
        self._x_var = x
        self._u_var = u
        self._K_lqr = -K_lqr  # so u = Kx convention if needed

        self.ocp = cp.Problem(cp.Minimize(cost), constr)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generic MPC for subsystems:
        Input/output are ABSOLUTE x/u, but optimization is in deltas around xs/us.
        """
        x0 = np.asarray(x0).reshape(-1)
        dx0 = x0 - self.xs
        self.dx0_var.value = dx0

        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            self.ocp.solve(warm_start=True, verbose=False)

        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # fallback: hold trim
            u0 = self.us.copy()
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        u_delta = self._u_var.value
        x_delta = self._x_var.value

        # Convert delta -> absolute
        u0 = u_delta[:, 0] + self.us
        x_traj = x_delta + self.xs.reshape(-1, 1)
        u_traj = u_delta + self.us.reshape(-1, 1)

        return u0, x_traj, u_traj
