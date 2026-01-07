import numpy as np
from MPCControl_base import MPCControl_base
from control import dlqr
from mpt4py import Polyhedron
import cvxpy as cp
import matplotlib.pyplot as plt


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        nx, nu, N = self.nx, self.nu, self.N
        A, B = self.A, self.B
        xs, us, = self.xs, self.us

        # ===== LQR feedback for tube =====
        Q = np.diag([20.0, 600.0, 30.0, 30.0])
        R = 0.1 * np.eye(1)

        K, Qf, _ = dlqr(A, B, Q, R)
        K = -K
        A_cl = A + B @ K
        self.K = K

        # ===== Set Constraint =====
        X, U = self.generate_constraint(xs, us, B)

        # ===== Generate Xf =====
        X_and_KU = X.intersect(Polyhedron.from_Hrep(U.A@K, U.b))
        Xf = self.max_invariant_set(A_cl, X_and_KU)

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
        # cost += 250000 * cp.sum_squares(v_var[1:] - v_var[:-1])

        constraints = []
        constraints.append(z_var[0] == x0_var)
        constraints.append(z_var[1:].T == A @ z_var[:-1].T + B @ v_var.T)
        constraints.append(U.A @ v_var.T <= U.b.reshape(-1, 1))
        constraints.append(X.A @ z_var[:-1].T <= X.b.reshape(-1, 1))
        constraints.append(Xf.A @ z_var[-1].T <= Xf.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        dxk = np.asarray(x0).reshape(-1) - np.asarray(self.xs).reshape(-1)
        self.x0_var.value = dxk
        self.ocp.solve(cp.OSQP, warm_start=True, max_iter=200000)
        assert self.ocp.status == cp.OPTIMAL, \
            f"The nominal y mpc solver returned status: {self.ocp.status}"

        dz = np.asarray(self.z_var[0].value).reshape(-1)
        dv = float(self.v_var[0].value)

        du = dv + float(self.K @ (dxk - dz))
        uy = du + float(self.us)

        return np.array([uy]), None, None

        # YOUR CODE HERE
        #################################################

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

        wy_min_phys, wy_max_phys = -150.0, 150.0
        alpha_min_phys, alpha_max_phys = -np.deg2rad(20), np.deg2rad(20)
        vy_min_phys, vy_max_phys = -150.0, 150.0
        y_min_phys, y_max_phys = -100.0, 100.0
        lower_X = np.array([
            wy_min_phys - xs[0],
            alpha_min_phys - xs[1],
            vy_min_phys - xs[2],
            y_min_phys - xs[3]
        ])
        upper_X = np.array([
            wy_max_phys - xs[0],
            alpha_max_phys - xs[1],
            vy_max_phys - xs[2],
            y_max_phys - xs[3]
        ])
        A_X = np.vstack((np.eye(4), -np.eye(4)))
        b_X = np.concatenate((upper_X, -lower_X))
        X = Polyhedron.from_Hrep(A=A_X, b=b_X)

        delta_phys, delta_max_phys = -np.deg2rad(14.99), np.deg2rad(14.99)
        delta_min = delta_phys - us[0]
        delta_max = delta_max_phys - us[0]
        A_U = np.vstack((np.eye(1), -np.eye(1)))
        b_U = np.array([delta_max, -delta_min])
        U = Polyhedron.from_Hrep(A=A_U, b=b_U)

        return X, U


