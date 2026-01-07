"""
Rocket dynamics (Python translation of MATLAB Rocket.m).

This module defines a Rocket class with the same core behavior as the
MATLAB implementation, including quaternion utilities and the dynamics
equations inside `dynamics_impl`.

Notes:
- This class does not depend on a Python RocketBase; instead it exposes
  the parameters normally provided by the base class as attributes with
  reasonable defaults. You can override them at construction time or by
  setting attributes directly before calling `dynamics_impl`.
- Quaternions follow the (w, x, y, z) convention.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

from src.rocket_base import *


# -----------------------------
# Rocket class (Python)
# -----------------------------
@dataclass
class Rocket(RocketBase):
    """Python port of MATLAB Rocket class.

    Attributes correspond to parameters typically provided by RocketBase
    in MATLAB. Defaults are placeholders; override with real values.
    """

    # Physical/environment parameters (override as needed)
    rho: float = 1.1589  # air density [kg/m^3]
    g: float = 9.806  # gravitational acceleration [m/s^2]
    mass: float = 2.0  # mass [kg]
    fuel_rate: float = 0.1
    fuel_consumption: float = 0.0
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0644, 0.0644, 0.0128]))  # inertia matrix [kg·m²]

    # Geometry/actuation parameters
    r_F: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.215]))  # thrust line offset along z [m]
    thrust_coeff: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # [a2, a1, a0]
    torque_coeff: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # [b2, b1, b0]

    model_params_filepath: Optional[str]
    Ts: float  # sampling time [s]

    controller_type: str = 'lmpc'


    def f(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute state derivative and outputs.

        Inputs:
        - x: 12-vector [w(3), r(3), v(3), p(4)] with r=(alpha, beta, gamma)
        - u: 4-vector [dR, dP, Pavg, Pdiff]

        Returns:
        - x_dot: 12-vector derivative
        - output: 3-vector [Va, alpha, beta]
        """

        # Decompose state
        w = x[0:3]
        r = x[3:6]
        v = x[6:9]

        b_F, b_M = self.getForceAndMomentFromThrust(u)

        # Angular acceleration
        w_dot = np.linalg.inv(self.J) @ (b_M - np.cross(w, self.J @ w))

        # Euler angle rates
        alp, bet, gam = r
        E_inv = (1/np.cos(bet)) * np.array(
            [
                [np.cos(gam), 				-np.sin(gam), 				0],
                [np.sin(gam)*np.cos(bet), 	np.cos(gam)*np.cos(bet), 	0],
                [-np.cos(gam)*np.sin(bet), 	np.sin(gam)*np.sin(bet), 	np.cos(bet)]
            ])
        r_dot = E_inv @ w

        # Linear acceleration and position
        Twb = self.eul2mat(r)
        v_dot = Twb @ (b_F / float(self.mass-self.fuel_consumption)) - np.array([0, 0, self.g])
        p_dot = v

        x_dot = np.concatenate([w_dot, r_dot, v_dot, p_dot])
        y = np.concatenate([b_F, b_M])  # for debugging
        return x_dot, y
        


    def getForceAndMomentFromThrust(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute thrust force and moment in body frame."""
        thrust = self.g * (self.thrust_coeff @ np.array([u[2]**2, u[2], 1.0]))
        torque = self.J[2,2] * (self.torque_coeff @ np.array([u[3]**2, u[3], 1.0]))

        b_eF = np.array([
            np.sin(u[1]),
            -np.sin(u[0]) * np.cos(u[1]),
            np.cos(u[0]) * np.cos(u[1])
        ])
        b_F = thrust * b_eF
        b_M = torque * b_eF + np.cross(self.r_F, b_F)
        return b_F, b_M
    

    def f_symbolic(self, x: ca.SX, u: ca.SX, mass: ca.SX=None) -> ca.SX:
        """Compute state derivative in casadi SX form.

        Inputs:
        - x: 12-vector [w(3), r(3), v(3), p(4)] with r=(alpha, beta, gamma)
        - u: 4-vector [dR, dP, Pavg, Pdiff]

        Returns:
        - x_dot: 12-vector derivative
        - y: 3-vector [Va, alpha, beta]
        """
        
        # Decompose state
        w = x[0:3]
        r = x[3:6]
        v = x[6:9]

        if mass is not None:
            self.mass = mass
        else:
            self.mass = ca.DM(self.mass)

        b_F, b_M = self.getForceAndMomentFromThrust_symbolic(u)

        # Angular acceleration
        J = ca.DM(self.J)
        Jinv = ca.DM(np.linalg.inv(self.J))
        w_dot = Jinv @ (b_M - ca.cross(w, J @ w))

        # Euler angle rates
        alp, bet, gam = r[0], r[1], r[2]
        E_inv = (1/ca.cos(bet)) * ca.vertcat(
            ca.horzcat(ca.cos(gam), -ca.sin(gam), 0),
            ca.horzcat(ca.sin(gam)*ca.cos(bet), ca.cos(gam)*ca.cos(bet), 0),
            ca.horzcat(-ca.cos(gam)*ca.sin(bet), ca.sin(gam)*ca.sin(bet), ca.cos(bet))
        )
        r_dot = E_inv @ w

        # Linear acceleration and position
        Twb = Rocket.eul2mat_symbolic(r)
        v_dot = Twb @ (b_F / self.mass) - ca.vertcat(0, 0, self.g)
        p_dot = v

        x_dot = ca.vertcat(w_dot, r_dot, v_dot, p_dot)
        y = ca.vertcat(b_F, b_M)  # for debugging
        return x_dot, y
        


    def getForceAndMomentFromThrust_symbolic(self, u: ca.SX) -> Tuple[ca.SX, ca.SX]:
        """Compute thrust force and moment in body frame."""
        thrust = self.g * (ca.DM(self.thrust_coeff.reshape(1,-1)) @ ca.vertcat(u[2]**2, u[2], 1.0))
        torque = self.J[2,2] * (ca.DM(self.torque_coeff.reshape(1,-1)) @ ca.vertcat(u[3]**2, u[3], 1.0))

        b_eF = ca.vertcat(
            ca.sin(u[1]),
            -ca.sin(u[0]) * ca.cos(u[1]),
            ca.cos(u[0]) * ca.cos(u[1])
        )
        b_F = thrust * b_eF
        b_M = torque * b_eF + ca.cross(self.r_F, b_F)
        return b_F, b_M
    

    def trim(self, x_ref: np.ndarray = np.zeros(12)) -> Tuple[np.ndarray, np.ndarray]:
        """Find steady-state (xs, us) such that f(xs, us) ≈ 0 using CasADi + Ipopt."""

        # Decision variables
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)

        # Objective: minimize squared dynamics residual
        xdot, _ = self.f_symbolic(x, u)
        obj = ca.sumsqr(xdot)

        # Bounds for states
        ubx = np.array([np.inf, np.inf, np.inf,
                        np.deg2rad(180), np.deg2rad(89), np.deg2rad(180),
                        np.inf, np.inf, np.inf,
                        np.inf, np.inf, np.inf])
        lbx = -ubx

        # Bounds for inputs
        uby = self.UBU
        lby = self.LBU

        lb = np.concatenate([lbx, lby])
        ub = np.concatenate([ubx, uby])

        # Initial guess: hover-ish throttle
        y0 = np.zeros(self.nx + self.nu)
        y0 = np.hstack([x_ref, np.zeros(self.nu)])	
        y0[self.nx + 2] = 60.0  # guess Pavg ≈ 60%

        # Build and solve NLP
        y = ca.vertcat(x, u)
        nlp = {"x": y, "f": obj}
        solver = ca.nlpsol("solver", "ipopt", nlp,
                    {"ipopt": {"print_level": 0}, "print_time": 0})

        res = solver(x0=y0, lbx=lb, ubx=ub)

        y_opt = np.array(res["x"]).flatten()
        self.xs = y_opt[:self.nx]
        self.us = y_opt[self.nx:]

        # Clean up small numerical noise
        self.xs[np.abs(self.xs) < 5e-2] = 0
        self.us[np.abs(self.us) < 1e-3] = 0

        return self.xs, self.us


    def linearize(self, xs: np.ndarray, us: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute linearization (A, B) around (xs, us) using CasADi."""

        # Decision variables
        x_symbolic = ca.SX.sym("x", self.nx)
        u_symbolic = ca.SX.sym("u", self.nu)

        # Dynamics
        x_dot, _ = self.f_symbolic(x_symbolic, u_symbolic)

        # Jacobians
        A = ca.jacobian(x_dot, x_symbolic)
        B = ca.jacobian(x_dot, u_symbolic)

        A_symbolic = ca.Function('sys_A', [x_symbolic, u_symbolic], [A])
        B_symbolic = ca.Function('sys_B', [x_symbolic, u_symbolic], [B])

        self.A_ss = np.array(A_symbolic(xs, us))
        self.B_ss = np.array(B_symbolic(xs, us))

        return self.A_ss, self.B_ss
    
    @staticmethod
    def decompose(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose the system matrices A, B into their respective components."""

        """Thrust vector angle dP to position x"""
        xids_x = np.array([1,4,6,9]); uids_x = np.array([1])
        xids_xi, xids_xj = np.meshgrid(xids_x,xids_x); uids_xi, uids_xj = np.meshgrid(xids_x,uids_x)
        Ax = A[xids_xi, xids_xj].T; Bx = B[uids_xi, uids_xj].T

        """Thrust vector angle dR to position y"""
        xids_y = np.array([0,3,7,10]); uids_y = np.array([0])
        xids_yi, xids_yj = np.meshgrid(xids_y,xids_y); uids_yi, uids_yj = np.meshgrid(xids_y,uids_y)
        Ay = A[xids_yi, xids_yj].T; By = B[uids_yi, uids_yj].T

        """Average throttle Pavg to height z"""
        xids_z = np.array([8,11]); uids_z = np.array([2])
        xids_zi, xids_zj = np.meshgrid(xids_z,xids_z); uids_zi, uids_zj = np.meshgrid(xids_z,uids_z)
        Az = A[xids_zi, xids_zj].T; Bz = B[uids_zi, uids_zj].T

        """Differential throttle to roll angle psi"""
        xids_p = np.array([2,5]); uids_p = np.array([3])
        xids_pi, xids_pj = np.meshgrid(xids_p,xids_p); uids_pi, uids_pj = np.meshgrid(xids_p,uids_p)
        Ap = A[xids_pi, xids_pj].T; Bp = B[uids_pi, uids_pj].T

        return Ax, Bx, Ay, By, Az, Bz, Ap, Bp

    def linearize_sys(self, xs: np.ndarray, us: np.ndarray):
        """Linearize nonlinear rocket dynamics at (xs, us) and return LTISys object."""
        x_sym = ca.SX.sym("x", self.nx)
        u_sym = ca.SX.sym("u", self.nu)

        x_dot, _ = self.f_symbolic(x_sym, u_sym)
        A = ca.jacobian(x_dot, x_sym)
        B = ca.jacobian(x_dot, u_sym)

        fA = ca.Function("A_func", [x_sym, u_sym], [A])
        fB = ca.Function("B_func", [x_sym, u_sym], [B])

        A_num = np.array(fA(xs, us))
        B_num = np.array(fB(xs, us))

        sys = LTISys(A_num, B_num, xs=xs, us=us)
        # self.sys = sys

        return sys

    @staticmethod
    def eul2mat(eul: np.ndarray) -> np.ndarray:
        """Rotation matrix from Euler angles (phi) body→world."""
        alp, bet, gam = eul

        def T1(a): return np.array([[1, 0, 0],
                                    [0, np.cos(a), np.sin(a)],
                                    [0, -np.sin(a), np.cos(a)]])

        def T2(a): return np.array([[np.cos(a), 0, -np.sin(a)],
                                    [0, 1, 0],
                                    [np.sin(a), 0, np.cos(a)]])

        def T3(a): return np.array([[np.cos(a), np.sin(a), 0],
                                    [-np.sin(a), np.cos(a), 0],
                                    [0, 0, 1]])

        return T1(-alp) @ T2(-bet) @ T3(-gam)
    
    @staticmethod
    def eul2mat_symbolic(eul: ca.SX) -> ca.SX:
        """Rotation matrix from Euler angles (phi) body→world."""
        alp, bet, gam = eul[0], eul[1], eul[2]

        def T1(a): return ca.vertcat(
                    ca.horzcat(1, 0, 0),
                    ca.horzcat(0, ca.cos(a), ca.sin(a)),
                    ca.horzcat(0, -ca.sin(a), ca.cos(a))
                )
        def T2(a): return ca.vertcat(
                    ca.horzcat(ca.cos(a), 0, -ca.sin(a)),
                    ca.horzcat(0, 1, 0),
                    ca.horzcat(ca.sin(a), 0, ca.cos(a))
                )
        def T3(a): return ca.vertcat(
                    ca.horzcat(ca.cos(a), ca.sin(a), 0),
                    ca.horzcat(-ca.sin(a), ca.cos(a), 0),
                    ca.horzcat(0, 0, 1)
                )

        return T1(-alp) @ T2(-bet) @ T3(-gam)
    

    def simulate(self, x0: np.ndarray, Tf: float, U: np.ndarray, method: str='nonlinear') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the system from initial state x0 under input u for Tf seconds.
        Uses zero-order hold on the input and simple Euler integration.
        Returns state and output trajectories (x, y).
        """
        N = int(Tf / self.Ts)  # number of simulation steps
        X = np.zeros((self.nx, N+1))
        T = np.arange(N) * self.Ts
        X[:, 0] = x0

        if U.ndim == 1:
            U = np.tile(U, (N, 1)).T
        elif U.shape[1] < N:
            raise ValueError("Input U has insufficient length")
        
        if method == 'linear':
            if not hasattr(self, 'xs') or not hasattr(self, 'us'):
                self.trim()
            if not hasattr(self, 'A_ss') or not hasattr(self, 'B_ss'):
                self.linearize(self.xs, self.us)
            for k in range(N):
                # Euler integration
                print(f"Simulating time {T[k]:.2f}", end=':')
                X[:, k+1] = Rocket.linear_integrate_step(self.A_ss, self.xs, self.B_ss, self.us, X[:, k], U[:, k], self.Ts)
                print('', end='\n')
        else:
            if method =='nonlinear_real':
                current_rocket = perturb_rocket(self)
            else:
                import copy
                current_rocket = copy.deepcopy(self)
            for k in range(N):
                # Euler integration
                print(f"Simulating time {T[k]:.2f}", end=':')
                U[:, k] = current_rocket.fuel_dynamics(U[:, k], self.Ts)
                X[:, k+1] = Rocket.integrate_step(current_rocket.f, X[:, k], U[:, k], self.Ts)
                print('', end='\n')
            
        return T, X[:,:-1], U
    
    
    def simulate_step(self, x0: np.ndarray, Ts: float, u0: np.ndarray, method: str='nonlinear') -> np.ndarray:
        if constraint_check(self, x0, u0):
            raise ValueError("Constraints violation detected, terminating...")

        u0 = self.fuel_dynamics(u0, Ts)
        if method == 'linear':
            if not hasattr(self, 'xs') or not hasattr(self, 'us'):
                self.trim()
            if not hasattr(self, 'A_ss') or not hasattr(self, 'B_ss'):
                self.linearize(self.xs, self.us)
            return Rocket.linear_integrate_step(self.A_ss, self.xs, self.B_ss, self.us, x0, u0, Ts)
        else:
            if method =='nonlinear_real':
                current_rocket = perturb_rocket(self)
            else:
                import copy
                current_rocket = copy.deepcopy(self)
            return Rocket.integrate_step(current_rocket.f, x0, u0, Ts)
        

    def simulate_control(self, mpc, sim_time: float, H: float, x0: np.ndarray, method: str='nonlinear', x_target=None, u_target=None, pos_control=None):
        N_cl = int(sim_time / self.Ts)
        N_ol = int(H / self.Ts)

        t_cl = np.zeros((N_cl+1))
        x_cl = np.zeros((12, N_cl+1))
        u_cl = np.zeros((4, N_cl))
        t_ol = np.zeros((N_ol+1, N_cl+1))
        x_ol = np.zeros((12, N_ol+1, N_cl+1))
        u_ol = np.zeros((4, N_ol, N_cl))

        x_cl[:, 0] = x0  # initial state

        if x_target is None:
            x_target = np.zeros((12, N_cl+1))
        elif x_target.ndim == 1:
            x_target = x_target.reshape(-1, 1).repeat(N_cl+1, axis=1)

        if method =='nonlinear_real':
            current_rocket = perturb_rocket(self)
            method = 'nonlinear'
        else:
            import copy
            current_rocket = copy.deepcopy(self)

        # Closed-loop simulation
        for k in range(N_cl):
            print(f"Simulating time {t_cl[k]:.2f}", end=': ')
            if pos_control is not None:
                x_target[6:9, k] = pos_control.get_u(x_cl[9:12, k])
            u_cl[:, k], x_ol[..., k], u_ol[..., k], t_ol[..., k] = mpc.get_u(t_cl[k], x_cl[:, k], x_target=x_target[:, k], u_target=u_target)
            x_cl[:, k+1] = current_rocket.simulate_step(x_cl[:, k], self.Ts, u_cl[:, k], method=method)
            mpc.estimate_parameters(x_cl[:, k:k+2], u_cl[:, k:k+1])
            t_cl[k+1] = t_cl[k] + self.Ts
            print('', end='\n')


        return t_cl, x_cl, u_cl, t_ol, x_ol, u_ol, x_target


    def fuel_dynamics(self, u: np.ndarray, Ts: float) -> np.ndarray:
        if self.fuel_consumption < self.mass/2:
            self.fuel_consumption += self.fuel_rate * Ts * u[2] / 100
        else:
            raise ValueError("Fuel exhausted")
            u[2] = 0.0; u[3] = 0.0
        if self.fuel_consumption > 0:
            print(f"Fuel left: {float(self.mass/2 - self.fuel_consumption):.2f} kg", end=', ')
        return u


    @staticmethod
    def integrate_step(f, x0, u, dt):
        """Integrate dynamics xdot = f(x,u) for dt seconds."""
        # ODE function with u fixed
        def dyn(t, x):
            xdot, _ = f(x, u)  # ignore output y
            return xdot

        sol = solve_ivp(dyn, [0, dt], x0, method="RK45", t_eval=[dt])
        return sol.y[:, -1]
    
    @staticmethod
    def linear_integrate_step(A, xs, B, us, x0, u, dt):
        """Integrate dynamics xdot = f(x,u) for dt seconds."""
        # ODE function with u fixed
        def dyn(t, x):
            xdot = A @ (x-xs) + B @ (u-us)  # ignore output y
            return xdot

        sol = solve_ivp(dyn, [0, dt], x0, method="RK45", t_eval=[dt])
        return sol.y[:, -1]
    
    def simulate_land(self, mpc, sim_time: float, H: float, x0: np.ndarray, method: str='nonlinear'):
        N_cl = int(sim_time / self.Ts)
        N_ol = int(H / self.Ts)

        t_cl = np.zeros((N_cl+1))
        x_cl = np.zeros((12, N_cl+1))
        u_cl = np.zeros((4, N_cl))
        t_ol = np.zeros((N_ol+1, N_cl+1))
        x_ol = np.zeros((12, N_ol+1, N_cl+1))
        u_ol = np.zeros((4, N_ol, N_cl))

        x_cl[:, 0] = x0  # initial state

        self.controller_type = mpc.__class__.__name__
        # print(self.controller_type)

        if method =='nonlinear_real':
            realistic_rocket = perturb_rocket(self)
            # Closed-loop simulation
            for k in range(N_cl):
                # flush not work due to cvxpy warning; tqdm need new package
                if k % 20 == 0:  # every 20 steps
                    print(f"Simulating time {t_cl[k]:.2f}")

                # setpoint_closed_loop[6:9, k] = setpoint_controller.get_u(x_traj_closed_loop[9:12, k])
                u_cl[:, k], x_ol[..., k], u_ol[..., k], t_ol[..., k] = mpc.get_u(t_cl[k], x_cl[:, k])
                x_cl[:, k+1] = realistic_rocket.simulate_step(x_cl[:, k], self.Ts, u_cl[:, k], method='nonlinear')
                t_cl[k+1] = t_cl[k] + self.Ts
        else:
            # Closed-loop simulation
            for k in range(N_cl):
                # flush not work due to cvxpy warning; tqdm need new package
                if k % 20 == 0:  # every 20 steps
                    print(f"Simulating time {t_cl[k]:.2f}")
                
                # setpoint_closed_loop[6:9, k] = setpoint_controller.get_u(x_traj_closed_loop[9:12, k])
                u_cl[:, k], x_ol[..., k], u_ol[..., k], t_ol[..., k] = mpc.get_u(t_cl[k], x_cl[:, k])
                x_cl[:, k+1] = self.simulate_step(x_cl[:, k], self.Ts, u_cl[:, k], method=method)
                t_cl[k+1] = t_cl[k] + self.Ts

        return t_cl, x_cl, u_cl, t_ol, x_ol, u_ol
    

    def add_noise(obj, flag_noise, T) -> np.ndarray:
        """Add noise sequence to the object."""
        noise = dict()
        noise["flag_noise"] = flag_noise
        w_min, w_max = -15.0, 5.0
        num_step = int(T / obj.Ts) + 10  # total time step
        rng = np.random.default_rng(24)
        match flag_noise:
            case 0:
                noise["w"] = np.zeros(num_step) # No noise
            case 1:
                noise["w"] = rng.uniform(w_min, w_max, size=num_step) # Uniform random noise
            case 2:
                noise["w"] = np.full(num_step, w_min) # Constant low noise                   
            case 3:
                # Step: half w_min, half w_max
                t1 = num_step // 2
                noise["w"] = np.concatenate([
                    np.full(t1, w_min),
                    np.full(num_step - t1, w_max)
                ])
            case _:
                raise ValueError("Invalid flag value")
        return noise["w"]	
    
    def simulate_step_dt(self, x, u, w, mpc):
        """use discrete-time model"""

        # Hard coding for sys_z and robust MPC only: changed in the future
        if constraint_check_sysZ(x, u):
            raise ValueError("Constraints violation detected, terminating...")
        xs, us = mpc.xs, mpc.us
        A, B = mpc.A, mpc.B
        x_next = A @ (x-xs) + B @ (u - us + w) + xs
        return x_next
    
    def simulate_subsystem(self, mpc, sim_time: float, x0: np.ndarray, w_type: str):
        """Simulation with one linearized subsystem only"""
        Ts = self.Ts
        match w_type:
            case "random":
                flag_noise = 1
            case "extreme":
                flag_noise = 2
            case "no_noise":
                flag_noise = 0
            case _:
                raise ValueError(f"Unknown w type: {w_type}")	
        w_traj_cl = self.add_noise(flag_noise, sim_time)

        # Varaibles for closed-loop recoding
        N_cl = int(sim_time / Ts)
        t_cl = Ts * np.arange(0, N_cl+1)
        x_cl = np.zeros((12, N_cl+1))
        u_cl = np.zeros((4, N_cl))
        x_cl[mpc.x_ids, 0] = x0[mpc.x_ids]  # initial state

        for k in range(N_cl):
            u_cl[mpc.u_ids, k], _, _ = mpc.get_u(x_cl[mpc.x_ids, k])
            x_cl[mpc.x_ids, k+1] = self.simulate_step_dt(x_cl[mpc.x_ids, k], u_cl[mpc.u_ids, k], w_traj_cl[k], mpc)
            t_cl[k+1] = t_cl[k] + Ts		

        return t_cl, x_cl, u_cl
    
def constraint_check_sysZ(x: np.ndarray, u: np.ndarray) -> bool:
    """Check if state x amd input u satisfy constraints for sys_z
    """
    LBU = np.array([40.0])  # [Pavg]
    UBU = np.array([80.0])  # [Pavg]

    LBX = np.array([-np.inf, 0.0]) # [vz, z]
    UBX = np.array([ np.inf, np.inf]) # # [vz, z]
    
    terminate = False
    if np.any(u < LBU-1e-5) or np.any(u > UBU+1e-5):
        print(f"Input violation: Pavg={u}, [LBU, UBU]={LBU, UBU}")
        terminate = True

    if np.any(x < LBX-1e-5) or np.any(x > UBX+1e-5):
        print(f"State violation: [vz,z]={x}, [LBX, UBX]={LBX, UBX}")
        terminate = True

    return terminate    
    

def perturb_rocket(rocket, seed: int | None = 1):
    """
    In order to mimic some realistic model mismatch between the MPC prediction model and the
    simulated 'reality',we implemente a reasonably different dynamics when you run nonlinear simulations.    
    Create a perturbed copy of `rocket` to act as the 'ground truth' model.
    Parameters are randomly biased to simulate mismatch/uncertainty.
    """    
    rng = np.random.default_rng(seed)

    def rand_plusminus_one(n: int) -> np.ndarray:
        # Uniform in [-1, 1], shape (n,)
        return rng.uniform(-1.0, 1.0, size=n)

    # Work on a deep copy so the original object is not modified
    import copy
    r = copy.deepcopy(rocket)

    # - parameter perturbations
    dMass = float(rand_plusminus_one(1)[0]) * 0.02
    dJ = (0.4 * r.J) @ np.diag(rand_plusminus_one(3))  # mirrors MATLAB: (0.4*J)*diag(rand)
    r.mass = r.mass + dMass
    r.J    = r.J + dJ    
    # - propeller model constants (hand-tuned, pseudo-realistic)
    r.thrust_coeff = np.array([1/22000, 1/45, 0.0], dtype=float) 
    r.torque_coeff = np.array([0.0, 0.0992, 0.0], dtype=float) 

    return r


def constraint_check(rocket, x: np.ndarray, u: np.ndarray) -> bool:
    """Check if state x amd input u satisfy constraints"""
    LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 40.0, -20.0])  # [dR, dP, Pavg, Pdiff]
    UBU = np.array([ np.deg2rad(15),  np.deg2rad(15), 80.0,  20.0])  # [dR, dP, Pavg, Pdiff]

    LBX = np.array([-np.inf, -np.inf, -np.inf,
                    -np.deg2rad(20), -np.deg2rad(20), -np.inf,
                    -np.inf, -np.inf, -np.inf,
                    -np.inf, -np.inf, 0.0]) # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
    UBX = np.array([np.inf,  np.inf,  np.inf,
                    np.deg2rad(20),  np.deg2rad(20),  np.inf,
                    np.inf,  np.inf,  np.inf,
                    np.inf,  np.inf, np.inf]) # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
    
    if rocket.controller_type == 'NmpcCtrl':
        LBU = np.array([-np.deg2rad(15), -np.deg2rad(15), 10.0, -20.0])  # [dR, dP, Pavg, Pdiff]
        UBU = np.array([ np.deg2rad(15),  np.deg2rad(15), 90.0,  20.0])  # [dR, dP, Pavg, Pdiff]  

        # |beta|<=80
        LBX = np.array([-np.inf, -np.inf, -np.inf,
                        -np.inf, -np.deg2rad(80), -np.inf,
                        -np.inf, -np.inf, -np.inf,
                        -np.inf, -np.inf, 0.0]) # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z] 
        UBX = np.array([np.inf,  np.inf,  np.inf,
                        np.inf,  np.deg2rad(80),  np.inf,
                        np.inf,  np.inf,  np.inf,
                        np.inf,  np.inf, np.inf]) # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
                  
    terminate = False
    if np.any(u < LBU-1e-5):
        for index in np.where(u < LBU-1e-5)[0]:
            print(f"\n Input {rocket.sys['InputName'][index]} violation: {u[index]:.2f} < {LBU[index]:.2f}", end=', ')
        terminate = True
    if np.any(u > UBU+1e-5):
        for index in np.where(u > UBU+1e-5)[0]:
            print(f"\n Input {rocket.sys['InputName'][index]} violation: {u[index]:.2f} > {UBU[index]:.2f}", end=', ')
        terminate = True
    if np.any(x < LBX-1e-5):
        for index in np.where(x < LBX-1e-5)[0]:
            print(f"\n State {rocket.sys['StateName'][index]} violation: {x[index]:.2f} < {LBX[index]:.2f}", end=', ')
    if np.any(x > UBX+1e-5):
        for index in np.where(x > UBX+1e-5)[0]:
            print(f"\n State {rocket.sys['StateName'][index]} violation: {x[index]:.2f} > {UBX[index]:.2f}", end=', ')

    return terminate


class LTISys:
    """Linearized state-space system with metadata."""

    def __init__(self, A, B, C=None, D=None, xs=None, us=None, name="complete system"):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.eye(A.shape[0]) if C is None else np.array(C)
        self.D = np.zeros((A.shape[0], B.shape[1])) if D is None else np.array(D)
        self.xs = np.array(xs) if xs is not None else None
        self.us = np.array(us) if us is not None else None
        self.name = name

        # Default naming
        self.state_name = [
            "wx", "wy", "wz",
            "alpha", "beta", "gamma",
            "vx", "vy", "vz",
            "x", "y", "z"
        ]
        self.state_unit = [
            "rad/s", "rad/s", "rad/s",
            "rad", "rad", "rad",
            "m/s", "m/s", "m/s",
            "m", "m", "m"
        ]

        self.input_name = ["d1", "d2", "Pavg", "Pdiff"]
        self.input_unit = ["rad", "rad", "%", "%"]

        self.output_name = self.state_name
        self.output_unit = self.state_unit

    # -----------------------
    #  Helper printing
    # -----------------------
    def info(self):
        """Print detailed information about the system (for students)."""
        print(f"\n System name: '{self.name}'")
        print("-" * 60)
        print(f"self.A shape: {self.A.shape}")
        print(f"self.B shape: {self.B.shape}")
        print(f"self.C shape: {self.C.shape}")
        print(f"self.D shape: {self.D.shape}")
        print(f"self.xs: {self.xs}")
        print(f"self.us: {self.us}")

        # optional attributes (only shown if defined)
        if hasattr(self, "idx"):
            print(f"self.idx: {self.idx}")
        if hasattr(self, "idu"):
            print(f"self.idu: {self.idu}")
        if hasattr(self, "idy"):
            print(f"self.idy: {self.idy}")

        print("\n🔹 State variables:")
        for n, u in zip(self.state_name, self.state_unit):
            print(f"   {n:>7s} [{u}]")
        print(f"→ access via self.state_name, self.state_unit")

        print("\n🔹 Input variables:")
        for n, u in zip(self.input_name, self.input_unit):
            print(f"   {n:>7s} [{u}]")
        print(f"→ access via self.input_name, self.input_unit")

        print("\n🔹 Output variables:")
        for n, u in zip(self.output_name, self.output_unit):
            print(f"   {n:>7s} [{u}]")
        print(f"→ access via self.output_name, self.output_unit")

        print("-" * 60)
        print(" To check data directly, try:")
        print("   sys.A, sys.B, sys.C, sys.D")
        print("   sys.xs, sys.us, sys.state_name, etc.")
        print("-" * 60)

    def __repr__(self):
        return f"<LTISys '{self.name}': A{self.A.shape}, B{self.B.shape}>"

    # -----------------------
    #  decomposition
    # -----------------------
    def _parse_subsystem(self, idx:np.ndarray, idu:np.ndarray, idy:np.ndarray, name:str):
        """Extract a subsystem based on index arrays."""

        # ensure all indices are numpy arrays
        idx = np.atleast_1d(np.array(idx, dtype=int))
        idu = np.atleast_1d(np.array(idu, dtype=int))
        idy = np.atleast_1d(np.array(idy, dtype=int))

        # submatrices
        A_sub = self.A[np.ix_(idx, idx)]
        B_sub = self.B[np.ix_(idx, idu)]
        C_sub = self.C[np.ix_(idy, idx)]

        # D matrix: sized by outputs × inputs
        ny, nu = len(idy), len(idu)
        D_sub = np.zeros((ny, nu))

        # create subsystem
        sub_sys = LTISys(
            A_sub,
            B_sub,
            C_sub,
            D_sub,
            xs=self.xs[idx] if self.xs is not None else None,
            us=self.us[idu] if self.us is not None else None,
            name=name,
        )

        # assign metadata
        sub_sys.state_name = [self.state_name[i] for i in idx]
        sub_sys.state_unit = [self.state_unit[i] for i in idx]
        sub_sys.input_name = [self.input_name[i] for i in idu]
        sub_sys.input_unit = [self.input_unit[i] for i in idu]
        sub_sys.output_name = [self.output_name[i] for i in idy]
        sub_sys.output_unit = [self.output_unit[i] for i in idy]

        sub_sys.idx = idx
        sub_sys.idu = idu
        sub_sys.idy = idy

        return sub_sys

    def decompose(self):
        """Split full system into four subsystems (sys_x, sys_y, sys_z, sys_roll)."""

        # use np.array directly for clarity and consistency
        sys_x = self._parse_subsystem(
            idx=np.array([1, 4, 6, 9]),  # wy, beta, vx, x
            idu=np.array([1]),           # d2
            idy=np.array([9]),           # x
            name="sys_x"
        )

        sys_y = self._parse_subsystem(
            idx=np.array([0, 3, 7, 10]),  # wx, alpha, vy, y
            idu=np.array([0]),            # d1
            idy=np.array([10]),           # y
            name="sys_y"
        )

        sys_z = self._parse_subsystem(
            idx=np.array([8, 11]),        # vz, z
            idu=np.array([2]),            # Pavg
            idy=np.array([11]),           # z
            name="sys_z"
        )

        sys_roll = self._parse_subsystem(
            idx=np.array([2, 5]),         # wz, gamma
            idu=np.array([3]),            # Pdiff
            idy=np.array([5]),            # gamma
            name="sys_roll"
        )

        return sys_x, sys_y, sys_z, sys_roll
    
    def _discretize(self, Ts: float):

        A_discrete, B_discrete, C_discrete, _, _ = cont2discrete(system=(self.A,self.B,self.C,self.D), dt=Ts)
        return A_discrete, B_discrete, C_discrete


__all__ = ["Rocket", "LTISys"]

