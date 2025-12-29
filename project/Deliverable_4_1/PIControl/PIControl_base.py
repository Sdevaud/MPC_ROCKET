from typing import Tuple

import cvxpy as cp
import numpy as np
from scipy.signal import cont2discrete


class PIControl_base:

    """Setpoint"""
    xs: np.ndarray

    """Controller gains"""
    Ts: float = 0.05

    """Integrator state"""
    integrator_state: float = 0.0


    """Control clamp"""
    u_min: float = -10.0
    u_max: float = 10.0


    def __init__(self, Kp: float, Ki: float, Ts: float, xs: float, u_min: float, u_max: float) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Ts = Ts
        self.xs = xs
        self.u_min = u_min
        self.u_max = u_max
    

    def compute_control(self, x0: float) -> float:
        """Compute the control input using PI control law.

        Args:
            x0 (float): Current state.

        Returns:
            float: Control input.
        """
        error = self.xs - x0
        u = self.Kp * error + self.integrator_state
        u_clamped = np.clip(u, self.u_min, self.u_max)
        self._update_integrator(x0, u)

        return u_clamped


    def _update_integrator(self, x0: float, u: float) -> None:
        if (u > self.u_min) and (u < self.u_max):
            self.integrator_state += self.Ki * (self.xs - x0) * self.Ts