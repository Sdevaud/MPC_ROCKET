import numpy as np
from mpt4py import Polyhedron
import cvxpy as cp
from control import dlqr
import matplotlib.pyplot as plt


from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])  # [wx, alpha, vy]
    u_ids: np.ndarray = np.array([0])        # delta1

    def _setup_controller(self) -> None:
        A, B = self.A, self.B
        N = self.N
        nx, nu = self.nx, self.nu

        Q = 10.0 * np.eye(nx)
        R = 1.0 * np.eye(nu)

        #Real space constraints
        #|delta1| <= 0.26
        M = np.array([[1.0], [-1.0]])
        m = np.array([0.26, 0.26])
        U_real = Polyhedron.from_Hrep(M, m)

        #|alpha| <= 0.1745
        F = np.array([[0.0, 1.0, 0.0],
                      [0.0, -1.0, 0.0]])
        f = np.array([0.1745, 0.1745])
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

        # Plotting the terminal set
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        O_inf.projection(dims=(0, 1)).plot(axs[0])
        axs[0].set_xlabel(r'$\Delta w_x$')
        axs[0].set_ylabel(r'$\Delta \alpha$')
        axs[0].set_title(r'Terminal Set Projection $(\Delta w_x,\Delta \alpha)$')

        O_inf.projection(dims=(0, 2)).plot(axs[1])
        axs[1].set_xlabel(r'$\Delta w_x$')
        axs[1].set_ylabel(r'$\Delta v_y$')
        axs[1].set_title(r'Terminal Set Projection $(\Delta w_x,\Delta v_y)$')

        O_inf.projection(dims=(1, 2)).plot(axs[2])
        axs[2].set_xlabel(r'$\Delta \alpha$')
        axs[2].set_ylabel(r'$\Delta v_y$')
        axs[2].set_title(r'Terminal Set Projection $(\Delta \alpha,\Delta v_y)$')

        plt.tight_layout()
        plt.show()


        #Variables in delta space
        self.dx_var = cp.Variable((nx, N + 1), name="dx")
        self.du_var = cp.Variable((nu, N), name="du")
        self.dx0_var = cp.Parameter((nx,), name="dx0")

        #Cost function in delta space
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
        #delta space solving
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

        #Convert back to real space from delta space
        x_traj = dx_traj + self.xs.reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1)
        u0 = u_traj[:, 0]
        return u0, x_traj, u_traj
