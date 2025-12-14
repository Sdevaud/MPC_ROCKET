import numpy as np
import cvxpy as cp
from mpt4py import Polyhedron
from control import dlqr

from .MPCControl_base import MPCControl_base


def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 200) -> Polyhedron:
    """
    Minimal robust positively invariant set for e^+ = A_cl e + w, w in W:
    E = sum_{i=0..inf} A_cl^i W
    Approximated by truncation until A_cl^i becomes small.
    """
    nx = A_cl.shape[0]
    E = W
    for i in range(1, max_iter):
        A_power = np.linalg.matrix_power(A_cl, i)
        E_next = E + (A_power @ W)
        E_next.minHrep()
        # stop when A^i is very small (geometric decay)
        if np.linalg.norm(A_power, 2) < 1e-3:
            E = E_next
            break
        E = E_next
    E.minHrep()
    return E


def max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 200) -> Polyhedron:
    """
    Maximal positively invariant set for x^+ = A_cl x inside X.
    O_{k+1} = pre(O_k) ∩ X, starting O_0 = X
    pre(O) = {x | A_cl x in O}
    """
    O = X
    for _ in range(max_iter):
        O_prev = O
        F, f = O.A, O.b
        # pre-set: F (A_cl x) <= f  => (F A_cl) x <= f
        O = Polyhedron.from_Hrep(np.vstack([F, F @ A_cl]), np.hstack([f, f]))
        O.minHrep()
        if O == O_prev:
            break
    return O


class MPCControl_z(MPCControl_base):
    """
    Part 6.1 robust tube MPC for sys_z:
      states: [vz, z]   (indices [8, 11])
      input : [Pavg]    (index [2])
    """
    x_ids: np.ndarray = np.array([8, 11])  # [vz, z]
    u_ids: np.ndarray = np.array([2])      # Pavg

    # tube objects
    K: np.ndarray
    Qf: np.ndarray
    E: Polyhedron
    X_tilde: Polyhedron
    U_tilde: Polyhedron
    Xf: Polyhedron

    def _setup_controller(self) -> None:
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu

        # ---------------------------
        # 1) Choose weights (tuneable)
        # ---------------------------
        # Goal: land to z_ref (via delta system -> 0) with settling <= 4s
        # Put strong weight on z, moderate on vz, moderate on du
        Q = np.diag([2.0, 20.0])     # [vz, z]
        R = np.diag([0.15])          # du = delta Pavg

        # LQR for local/tube feedback K and terminal cost Qf
        K_lqr, Qf, _ = dlqr(A, B, Q, R)
        self.K = -K_lqr  # dlqr returns stabilizing for u = -Kx, we use u = Kx
        self.Qf = Qf

        A_cl = A + B @ self.K

        # ---------------------------
        # 2) Constraints (ABSOLUTE -> DELTA)
        # ---------------------------
        # Input limits from PDF: Pavg in [10, 90]
        Pmin, Pmax = 10.0, 90.0
        du_min = Pmin - float(self.us)
        du_max = Pmax - float(self.us)
        U = Polyhedron.from_Hrep(
            A=np.array([[1.0], [-1.0]]),
            b=np.array([du_max, -du_min]),
        )

        # State constraints:
        # - enforce z >= 0 (PDF eq. 9)
        # - keep vz reasonable (helps feasibility / realism)
        # - optional upper z bound (keeps set bounded for invariant computations)
        vz_max = 20.0
        z_min = 0.0
        z_max = 30.0

        vzs = float(self.xs[0])  # xs is sliced already by MPCControl_base
        zs  = float(self.xs[1])

        # constraints on dx: x = dx + xs
        # vz <= vz_max  => dx_vz <= vz_max - vzs
        # vz >= -vz_max => -dx_vz <= vz_max + vzs
        # z >= 0        => -(dx_z + zs) <= 0 => -dx_z <= zs
        # z <= z_max    => dx_z <= z_max - zs
        X_A = np.array([
            [ 1.0,  0.0],
            [-1.0,  0.0],
            [ 0.0,  1.0],
            [ 0.0, -1.0],
        ])
        X_b = np.array([
            vz_max - vzs,
            vz_max + vzs,
            z_max - zs,
            zs - z_min,   # since -dx_z <= zs - z_min
        ])
        X = Polyhedron.from_Hrep(X_A, X_b)
        X.minHrep()

        # ---------------------------
        # 3) Disturbance set W (PDF eq. 10)
        #    dx+ = A dx + B du + B w, w in [-15, 5]
        # ---------------------------
        w_lo, w_hi = -15.0, 5.0
        W1 = Polyhedron.from_Hrep(
            A=np.array([[1.0], [-1.0]]),
            b=np.array([w_hi, -w_lo]),
        )
        # W in state space is B * w
        W = W1.affine_map(B)  # 2D polyhedron in dx-space
        W.minHrep()

        # ---------------------------
        # 4) Tube: E, tightened constraints, terminal set
        # ---------------------------
        self.E = min_robust_invariant_set(A_cl, W)
        self.E.minHrep()

        # Tightened sets
        self.X_tilde = X - self.E
        self.X_tilde.minHrep()

        KE = self.E.affine_map(self.K)  # K*E in input space
        KE.minHrep()
        self.U_tilde = U - KE
        self.U_tilde.minHrep()

        # Terminal set: invariant for A_cl within tightened constraints AND tightened input
        # Need: x in X_tilde and Kx in U_tilde  (in delta-space)
        X_and_KU = self.X_tilde.intersect(Polyhedron.from_Hrep(self.U_tilde.A @ self.K, self.U_tilde.b))
        X_and_KU.minHrep()
        self.Xf = max_invariant_set(A_cl, X_and_KU)
        self.Xf.minHrep()

        # ---------------------------
        # 5) Nominal MPC (tightened) with tube initial constraint
        # ---------------------------
        self.xbar = cp.Variable((nx, N + 1), name="xbar")   # nominal delta state
        self.ubar = cp.Variable((nu, N), name="ubar")       # nominal delta input

        self.dx0_var = cp.Parameter((nx,), name="dx0")

        cost = 0
        constraints = []

        # Tube initial condition: dx0 - xbar0 ∈ E
        constraints += [self.E.A @ (self.dx0_var - self.xbar[:, 0]) <= self.E.b]

        # Dynamics + constraints
        for k in range(N):
            cost += cp.quad_form(self.xbar[:, k], Q)
            cost += cp.quad_form(self.ubar[:, k], R)

            constraints += [self.xbar[:, k + 1] == A @ self.xbar[:, k] + B @ self.ubar[:, k]]
            constraints += [self.X_tilde.A @ self.xbar[:, k] <= self.X_tilde.b]
            constraints += [self.U_tilde.A @ self.ubar[:, k] <= self.U_tilde.b]

        # Terminal
        cost += cp.quad_form(self.xbar[:, N], self.Qf)
        constraints += [self.Xf.A @ self.xbar[:, N] <= self.Xf.b]

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None):
        """
        sys_z robust tube MPC
        x0 is already sliced to [vz, z]
        returns ABSOLUTE input Pavg (shape (1,))
        """
        x0 = np.asarray(x0).reshape(-1)

        # delta initial condition
        dx0 = x0 - self.xs
        self.dx0_var.value = dx0

        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            self.ocp.solve(warm_start=True, verbose=False)

        # fallback if infeasible
        if self.ocp.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            u_abs = float(np.clip(float(self.us), 10.0, 90.0))
            u0 = np.array([u_abs], dtype=float)
            x_traj = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(u0.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj

        xbar = self.xbar.value
        ubar = self.ubar.value

        # tube feedback
        e0 = dx0 - xbar[:, 0]                         # e0 in E
        du0 = float(ubar[:, 0] + (self.K @ e0))       # delta input

        # absolute input + enforce bounds from Deliverable 6
        u_abs = float(du0 + float(self.us))
        u_abs = float(np.clip(u_abs, 10.0, 90.0))
        u0 = np.array([u_abs], dtype=float)

        # predicted nominal trajectories (for plotting/debug)
        x_traj = xbar + self.xs.reshape(-1, 1)
        u_traj = ubar + float(self.us)

        return u0, x_traj, u_traj