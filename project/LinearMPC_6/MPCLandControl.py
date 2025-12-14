import numpy as np

from src.rocket import Rocket

from .MPCControl_roll import MPCControl_roll
from .MPCControl_xvel import MPCControl_xvel
from .MPCControl_yvel import MPCControl_yvel
from .MPCControl_zvel import MPCControl_zvel


class MPCLandControl:
    mpc_x: MPCControl_x
    mpc_y: MPCControl_y
    mpc_z: MPCControl_z
    mpc_roll: MPCControl_roll

    def __init__(self) -> None:
        pass

    def new_controller(self, rocket: Rocket, Ts: float, H: float, x_ref: np.ndarray) -> None:
        self.xs, self.us = rocket.trim(x_ref)
        A, B = rocket.linearize(self.xs, self.us)

        self.mpc_x = MPCControl_x(A, B, self.xs, self.us, Ts, H)
        self.mpc_y = MPCControl_y(A, B, self.xs, self.us, Ts, H)
        self.mpc_z = MPCControl_z(A, B, self.xs, self.us, Ts, H)
        self.mpc_roll = MPCControl_roll(A, B, self.xs, self.us, Ts, H)

        return self


    def get_u(
        self,
        t0: float,
        x0: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u0 = np.zeros(4)
        t_traj = np.arange(self.mpc_x.N + 1) * self.mpc_x.Ts + t0
        x_traj = np.zeros((12, self.mpc_x.N + 1))
        u_traj = np.zeros((4, self.mpc_x.N))

        if x_target is None:
            x_target = self.xs

        if u_target is None:
            u_target = self.us

        u0[self.mpc_x.u_ids], x_traj[self.mpc_x.x_ids], u_traj[self.mpc_x.u_ids] = (
            self.mpc_x.get_u(
                x0[self.mpc_x.x_ids],
                x_target[self.mpc_x.x_ids],
                u_target[self.mpc_x.u_ids],
            )
        )
        u0[self.mpc_y.u_ids], x_traj[self.mpc_y.x_ids], u_traj[self.mpc_y.u_ids] = (
            self.mpc_y.get_u(
                x0[self.mpc_y.x_ids],
                x_target[self.mpc_y.x_ids],
                u_target[self.mpc_y.u_ids],
            )
        )
        u0[self.mpc_z.u_ids], x_traj[self.mpc_z.x_ids], u_traj[self.mpc_z.u_ids] = (
            self.mpc_z.get_u(
                x0[self.mpc_z.x_ids],
                x_target[self.mpc_z.x_ids],
                u_target[self.mpc_z.u_ids],
            )
        )
        (
            u0[self.mpc_roll.u_ids],
            x_traj[self.mpc_roll.x_ids],
            u_traj[self.mpc_roll.u_ids],
        ) = self.mpc_roll.get_u(
            x0[self.mpc_roll.x_ids],
            x_target[self.mpc_roll.x_ids],
            u_target[self.mpc_roll.u_ids],
        )

        return u0, x_traj, u_traj, t_traj
