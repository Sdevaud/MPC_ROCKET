import numpy as np
from MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron
import cvxpy as cp
import matplotlib.pyplot as plt

class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])


    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        A = self.A
        B = self.B
        nx, nu, N = self.nx, self.nu, self.N
        Q = np.diag([1.0, 1.0])
        R = np.diag([10])
        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        A_cl = A + B @ K

        X, U, W_u = self.generate_constraint()
        W_x = W_u.affine_map(B)
        E = self.min_robust_invariant_set(A_cl, W_x)
        X_tilde = X - E
        KE = E.affine_map(K)
        U_tilde = U - KE
        X_tilde_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ K, U_tilde.b))
        Xf_tilde = self.max_invariant_set(A_cl, X_tilde_and_KU_tilde)

        print("X_tilde empty?", X_tilde.is_empty)
        print("U_tilde empty?", U_tilde.is_empty)
        print("Xf_tilde empty?", Xf_tilde.is_empty)
        print("E subset X ?", X.contains(E))

        z_var = cp.Variable((N+1, nx), name='z')
        v_var = cp.Variable((N, nu), name='v')
        x0_var = cp.Parameter((nx,), name='x0')

        fig_v1, ax_v1 = plt.subplots(1, 1)

        X.plot(ax_v1, color='g', opacity=0.2, label='X_vis')
        # X_tilde.plot(ax_v1, color='y', opacity=0.4, label='X_tilde_vis')
        E.plot(ax_v1, color='r', opacity=0.5, label='E')

        plt.legend()
        plt.title("Visualization of X_vis, X_tilde_vis and E")
        plt.show()


        ##### DEBUG #######

        cost = 0
        for i in range(N):
            cost += cp.quad_form(z_var[i], Q)
            cost += cp.quad_form(v_var[i], R)
        cost += cp.quad_form(z_var[-1], Qf)

        constraints = []
        constraints.append(E.A @ (x0_var - z_var[0]) <= E.b)
        constraints.append(z_var[1:].T == A @ z_var[:-1].T + B @ v_var.T)
        constraints.append(X_tilde.A @ z_var[:-1].T <= X_tilde.b.reshape(-1, 1))
        constraints.append(U_tilde.A @ v_var[:-1].T <= U_tilde.b.reshape(-1, 1))
        constraints.append(Xf_tilde.A @ z_var[-1].T <= Xf_tilde.b.reshape(-1, 1))

        self.z_var = z_var
        self.v_var = v_var
        self.x0_var = x0_var
        self.K = K
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x0_var.value = x0
        self.ocp.solve(solver=cp.OSQP, warm_start=True)
        print("Status:", self.ocp.status)
        print("z0:", self.z_var[0].value)
        print("v0:", self.v_var[0].value)
        z0 = self.z_var[0].value
        v0 = self.v_var[0].value
        u0 = v0 + self.K @ (x0 - z0)
        #################################################
        return u0, None, None

    @staticmethod
    def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 50) -> Polyhedron:
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
            O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
            O.minHrep()
            if O == Oprev:
                converged = True
                break
            itr += 1
        return O
    
    @staticmethod
    def generate_constraint():
        # ----- X : contrainte forte sur z (2e état), bornes larges sur v_z -----
        z_min, z_max = 0.0, 50.0
        v_min, v_max = -10.0, 10.0   # bornes larges fictives

        # ordre des états : [v_z, z]
        lower_X = np.array([v_min, z_min])
        upper_X = np.array([v_max, z_max])

        A_X = np.vstack((np.eye(2), -np.eye(2)))
        b_X = np.concatenate((upper_X, -lower_X))
        X   = Polyhedron.from_Hrep(A=A_X, b=b_X)

        # ----- U : Pavg ∈ [Pavg_min, Pavg_max] -----
        Pavg_min, Pavg_max = 40.0, 80.0
        A_U = np.vstack((np.eye(1), -np.eye(1)))
        b_U = np.array([Pavg_max, -Pavg_min])
        U   = Polyhedron.from_Hrep(A=A_U, b=b_U)

        # ----- W_u : disturbance d'entrée w ∈ [w_min, w_max] -----
        w_min, w_max = -15.0, 5.0
        A_Wu = np.vstack((np.eye(1), -np.eye(1)))    # [[1],[-1]]
        b_Wu = np.array([w_max, -w_min])             # [5, 15]
        W_u  = Polyhedron.from_Hrep(A=A_Wu, b=b_Wu)

        return X, U, W_u




