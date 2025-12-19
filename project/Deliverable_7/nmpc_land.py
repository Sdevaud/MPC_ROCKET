import numpy as np
import casadi as ca
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """



    def __init__(self, rocket, Ts, xs, us, N):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.rocket = rocket
        self.Ts = Ts
        self.N = N
        self.nx = np.size(xs)
        self.nu = np.size(us)
        self.xs = xs
        self.us = us

        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]

        self._setup_controller()

    def _setup_controller(self) -> None:

        nx = self.nx
        nu = self.nu
        N = self.N
        Ts = self.Ts

        opti = ca.Opti()

        self.x0_param = opti.parameter(nx)
        self.X = opti.variable(nx, N + 1)
        self.U = opti.variable(nu, N)

        # ---- initial condition ----
        opti.subject_to(self.X[:, 0] == self.x0_param)

        # ---- constants ----
        beta_max = np.deg2rad(80.0)
        delta_max = np.deg2rad(15.0)

        # ---- dynamics + constraints ----
        for k in range(N):

            # Euler discretization
            x_next = self.X[:, k] + Ts * self.f(self.X[:, k], self.U[:, k])
            opti.subject_to(self.X[:, k + 1] == x_next)

            # satate constraint |z| > 0 and |beta| < 80 deg
            opti.subject_to(self.X[11, k] >= 0)
            opti.subject_to(opti.bounded(-beta_max, self.X[4, k], beta_max))

            # input constraints
            opti.subject_to(opti.bounded(-delta_max, self.U[0, k], delta_max))
            opti.subject_to(opti.bounded(-delta_max, self.U[1, k], delta_max))
            opti.subject_to(opti.bounded(10.0, self.U[2, k], 90.0))
            opti.subject_to(opti.bounded(-20.0, self.U[3, k], 20.0))

        # terminal constraints
        opti.subject_to(self.X[11, N] >= 0)
        opti.subject_to(opti.bounded(-beta_max, self.X[4, N], beta_max))

        # ---- cost function ----
        Q = np.diag([
            10, 10, 10,          # angular rates
            10, 10, 100,        # angles
            10, 10, 10,         # velocities
            70, 70, 100       # positions
        ])

        R = np.diag([
            50, 50,             # gimbal angles
            0.1,              # average thrust
            0.1 
        ])

        P = Q  # terminal cost (simple but effective)

        cost = 0
        for k in range(N):
            dx = self.X[:, k] - self.xs
            du = self.U[:, k] - self.us
            cost += ca.mtimes([dx.T, Q, dx])
            cost += ca.mtimes([du.T, R, du])

        dxN = self.X[:, N] - self.xs
        cost += ca.mtimes([dxN.T, P, dxN])

        opti.minimize(cost)

        # ---- solver ----
        opti.solver(
            "ipopt",
            {"expand": True}
        )
        
        self.ocp = opti

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # set initial state
        self.ocp.set_value(self.x0_param, x0)

        # solve NMPC
        sol = self.ocp.solve()

        # extract open-loop solution
        x_ol = sol.value(self.X)
        u_ol = sol.value(self.U)
        u0 = u_ol[:, 0]

        # time vector
        t_ol = t0 + self.Ts * np.arange(self.N + 1)

        return u0, x_ol, u_ol, t_ol