import numpy as np
from MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron
import cvxpy as cp
import matplotlib.pyplot as plt


class MPCControl_z(MPCControl_base):
    x_ids = np.array([8, 11])   # [vz, z]
    u_ids = np.array([2])       # thrust

    def _setup_controller(self) -> None:

        nx, nu, N = self.nx, self.nu, self.N
        A, B = self.A, self.B
        xs, us, = self.xs, self.us

        # ===== LQR feedback for tube =====
        Q = np.diag([12.0, 20.0])
        R = 0.5 * np.eye(1)

        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        A_cl = A + B @ K
        self.K = K

        # ===== eigen value of Acl =====
        eigvals_Acl = np.linalg.eigvals(A_cl)
        print("norm of biggest eigen value of Acl :")
        print(np.max(np.abs(eigvals_Acl)))

        # ===== Set Constraint =====
        X, U, W = self.generate_constraint(xs, us, B)
        self.W = W

        # ===== Generate Set =====
        E = self.min_robust_invariant_set(A_cl, W)
        X_tilde = X - E
        KE = E.affine_map(K)
        U_tilde = U - KE

        # ===== Generate Xf =====
        X_and_KU = X.intersect(Polyhedron.from_Hrep(U.A@K, U.b))
        Xf = self.max_invariant_set(A_cl, X_and_KU)

        X_tilde_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A@K, U_tilde.b))
        Xf_tilde = self.max_invariant_set(A_cl, X_tilde_and_KU_tilde)

        # ===== Build cost Function =====
        z_var = cp.Variable((N+1, nx), name='z')
        v_var = cp.Variable((N, nu), name='v')
        x0_var = cp.Parameter((nx,), name='x0')

        self.z_var = z_var
        self.v_var = v_var
        self.x0_var = x0_var

        cost = 0
        for i in range(N):
            cost += cp.quad_form(z_var[i], Q)
            cost += cp.quad_form(v_var[i], R)
        cost += cp.quad_form(z_var[-1], Qf)

        constraints = []
        constraints.append(E.A @ (x0_var - z_var[0]) <= E.b)
        constraints.append(z_var[1:].T == A @ z_var[:-1].T + B @ v_var.T)
        constraints.append(X_tilde.A @ z_var[:-1].T <= X_tilde.b.reshape(-1, 1))
        constraints.append(U_tilde.A @ v_var.T <= U_tilde.b.reshape(-1, 1))
        constraints.append(Xf_tilde.A @ z_var[-1].T <= Xf_tilde.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # ===== Visualisation =====

        print("\n========== INCLUSIONS ==========")
        print("E c X ?", X.contains(E))
        print("KE c U ?", U.contains(KE))
        print("Xf_tilde c X_tilde ?", X_tilde.contains(Xf_tilde))
        print("X_tilde empty ?", X_tilde.is_empty)
        print("U_tilde empty ?", U_tilde.is_empty)
        print("Xf_tilde empty ?", Xf_tilde.is_empty)

        fig2, ax2 = plt.subplots(1, 1)
        X.plot(ax2, color='g', opacity=0.5, label=r'$\mathcal{X}$')
        X_tilde.plot(ax2, color='y', opacity=0.5, label=r'$\mathcal{\tilde{X}}$')
        E.plot(ax2, color='r', opacity=0.5, label=r'$\mathcal{E}$')
        plt.legend()
        plt.show()

        fig3, ax3 = plt.subplots(1, 1)
        Xf.plot(ax3, color='g', opacity=0.5, label=r'$\mathcal{X}_f$')
        Xf_tilde.plot(ax3, color='b', opacity=0.5, label=r'$\tilde{\mathcal{X}}_f$')
        plt.legend()
        plt.show()


    def get_u(
        self, x0: np.ndarray, x_target=None, u_target=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        dxk = np.asarray(x0).reshape(-1) - np.asarray(self.xs).reshape(-1)
        self.x0_var.value = dxk
        self.ocp.solve(cp.OSQP, warm_start=True, max_iter=200000)
        assert self.ocp.status == cp.OPTIMAL, \
            f"The tube mpc solver returned status: {self.ocp.status}"

        dz = np.asarray(self.z_var[0].value).reshape(-1)
        dv = float(self.v_var[0].value)

        du = dv + float(self.K @ (dxk - dz))
        uz = du + float(self.us)

        return np.array([uz]), None, None

    
    @staticmethod
    def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 30) -> Polyhedron:
        nx = A_cl.shape[0]
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep()
            if np.linalg.norm(A_cl_ith_power, ord=2) < 1e-2:
                break
            Omega = Omega_next
            itr += 1
        return Omega_next

    @staticmethod
    def max_invariant_set(A_cl, X: Polyhedron, max_iter=50) -> Polyhedron:
        O = X
        itr = 1
        converged = False
        while itr < max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O.minHrep()
            if O == Oprev:
                converged = True
                break
            itr += 1
        
        if converged:
            print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
        return O
    
    @staticmethod
    def generate_constraint(xs, us, B):
        xs = np.asarray(xs).reshape(-1)
        us = np.asarray(us).reshape(-1)
        B  = np.asarray(B).reshape(-1, 1)

        v_min_phys, v_max_phys = -10.0, 10.0
        z_min_phys, z_max_phys = 0.0, 12.0
        lower_X = np.array([
            v_min_phys - xs[0],
            z_min_phys - xs[1]
        ])
        upper_X = np.array([
            v_max_phys - xs[0],
            z_max_phys - xs[1]
        ])
        A_X = np.vstack((np.eye(2), -np.eye(2)))
        b_X = np.concatenate((upper_X, -lower_X))
        X = Polyhedron.from_Hrep(A=A_X, b=b_X)

        Pavg_min_phys, Pavg_max_phys = 40.05, 79.95
        Pavg_min = Pavg_min_phys - us[0]
        Pavg_max = Pavg_max_phys - us[0]
        A_U = np.vstack((np.eye(1), -np.eye(1)))
        b_U = np.array([Pavg_max, -Pavg_min])
        U = Polyhedron.from_Hrep(A=A_U, b=b_U)

        w_min_u, w_max_u = -15.0, 5.0
        W_u = Polyhedron.from_Hrep(
            A=np.vstack((np.eye(1), -np.eye(1))),
            b=np.array([w_max_u, -w_min_u])
        )
        W = W_u.affine_map(B)

        return X, U, W

