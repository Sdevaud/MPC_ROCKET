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

        X, U, W_x = self.generate_constraint(B)
        E = self.min_robust_invariant_set(A_cl, W_x)
        X_tilde = X - E
        KE = E.affine_map(K)
        U_tilde = U - KE
        X_tilde_and_KU_tilde = X_tilde.intersect(Polyhedron.from_Hrep(U_tilde.A @ K, U_tilde.b))
        print("x_tilde_and_KU_tilde empty ?", X_tilde_and_KU_tilde.is_empty)

        Xf_tilde = self.max_invariant_set(A_cl, X_tilde_and_KU_tilde)

        z_var = cp.Variable((N+1, nx), name='z')
        v_var = cp.Variable((N, nu), name='v')
        x0_var = cp.Parameter((nx,), name='x0')

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

        ##### DEBUG ##########################################################

        # print("\n========== DEBUG : MATRICES ==========")
        # print("A =\n", A)
        # print("B =\n", B)
        # print("K =\n", K)
        # print("A_cl = A + B K =\n", A_cl)
        # print("Spectral radius A_cl:", max(abs(np.linalg.eigvals(A_cl))))

        # print("\n========== DEBUG : DIMENSIONS ==========")
        # print("nx =", nx, "nu =", nu, "N =", N)
        # print("A.shape =", A.shape)
        # print("B.shape =", B.shape)
        # print("K.shape =", K.shape)

        # def debug_poly(name, P: Polyhedron):
        #     print(f"\n--- {name} ---")
        #     print("A shape:", P.A.shape)
        #     print("b shape:", P.b.shape)
        #     print("A =\n", P.A)
        #     print("b =\n", P.b)
        #     print("is empty ?", P.is_empty)

        # print("\n========== DEBUG : POLYHEDRA ==========")
        # debug_poly("X", X)
        # debug_poly("U", U)
        # debug_poly("W_x = B W_u", W_x)
        # print("\nB =\n", B)
        # print("Worst-case disturbance on x:")
        # print("min B*w =", B * (-15))
        # print("max B*w =", B * (5))
        # debug_poly("E (mRPI)", E)
        # debug_poly("X_tilde = X - E", X_tilde)
        # debug_poly("U_tilde = U - K E", U_tilde)
        # debug_poly("Xf_tilde", Xf_tilde)

        # print("\n========== DEBUG : INCLUSIONS ==========")
        # print("E ⊂ X ?", X.contains(E))
        # print("KE ⊂ U ?", U.contains(KE))
        # print("Xf_tilde ⊂ X_tilde ?", X_tilde.contains(Xf_tilde))

        # print("\n========== DEBUG : FEASIBILITY ==========")
        # print("X_tilde empty ?", X_tilde.is_empty)
        # print("U_tilde empty ?", U_tilde.is_empty)
        # print("Xf_tilde empty ?", Xf_tilde.is_empty)

        # print("\n========== DEBUG : VISUALISATION ==========")
        # fig, ax = plt.subplots(1, 1)
        # X.plot(ax=ax, color='green', opacity=0.15, label='X')
        # E.plot(ax=ax, color='red', opacity=0.4, label='E')
        # X_tilde.plot(ax=ax, color='blue', opacity=0.25, label='X_tilde')
        # # Xf_tilde.plot(ax=ax, color='purple', opacity=0.4, label='Xf_tilde')
        # ax.set_title("State constraint sets")
        # ax.legend()
        # plt.grid(True)
        # plt.show()

        # print("\n========== DEBUG : INPUT SETS (1D) ==========")
        # print("U bounds:")
        # print("  u <=", U.b[0])
        # print("  u >=", -U.b[1])

        # print("\nU_tilde bounds:")
        # print("  u <=", U_tilde.b[0])
        # print("  u >=", -U_tilde.b[1])

        # print("\n========== DEBUG : CVXPY CHECK ==========")
        # print("z_var shape:", z_var.shape)
        # print("v_var shape:", v_var.shape)
        # print("x0_var shape:", x0_var.shape)

        # print("\n========== END DEBUG =====================================")


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        # 1) steady-state associé à la consigne
        xs, us = self.steady_state(self.x_ref)

        # 2) coordonnées delta
        x0_delta = x0 - xs

        # 3) résoudre le MPC
        self.x0_var.value = x0_delta
        self.ocp.solve(solver=cp.OSQP, warm_start=True)

        if self.ocp.status != cp.OPTIMAL:
            raise RuntimeError("MPC infeasible")

        z0 = self.z_var[0].value
        v0 = self.v_var[0].value

        # 4) loi tube MPC + recentrage physique
        u0 = us + v0 + self.K @ (x0_delta - z0)


        #################################################
        return u0, None, None

    @staticmethod
    def min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 15) -> Polyhedron:
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
    def max_invariant_set(A_cl, X: Polyhedron, max_iter=15) -> Polyhedron:
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
            print('Iteration {0}... not yet converged'.format(itr))
            itr += 1
        
        if converged:
            print('Maximum invariant set successfully computed after {0} iterations.\n'.format(itr))
        return O
    
    @staticmethod
    def steady_state(self, x_ref):
        A, B = self.A, self.B
        nx = A.shape[0]

        xs = x_ref.reshape(-1, 1)
        us = np.linalg.pinv(B) @ ((np.eye(nx) - A) @ xs)

        return xs.flatten(), us.flatten()

    
    @staticmethod
    def generate_constraint(B=None):
        z_min, z_max = -12, 12
        v_min, v_max = -2, 2
        lower_X = np.array([v_min, z_min])
        upper_X = np.array([v_max, z_max])
        A_X = np.vstack((np.eye(2), -np.eye(2)))
        b_X = np.concatenate((upper_X, -lower_X))
        X = Polyhedron.from_Hrep(A=A_X, b=b_X)

        Pavg_min, Pavg_max = -20, 20
        A_U = np.vstack((np.eye(1), -np.eye(1)))
        b_U = np.array([Pavg_max, -Pavg_min])
        U = Polyhedron.from_Hrep(A=A_U, b=b_U)

        w1_min, w1_max = -10*0.00865235, 10*0.00865235
        w2_min, w2_max = -10*0.00021631, 10*0.00021631

        lower_W = np.array([w1_min, w2_min])
        upper_W = np.array([w1_max, w2_max])
        A_W = np.vstack((np.eye(2), -np.eye(2)))
        b_W = np.concatenate((upper_W, -lower_W))
        W = Polyhedron.from_Hrep(A=A_W, b=b_W)

        return X, U, W
