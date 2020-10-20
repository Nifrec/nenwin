"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Settings (constants) for the test cases, such as required numerical precision,
and auxiliary functions.
"""
import numpy as np
from typing import Tuple

NUMERICAL_ABS_ACCURACY_REQUIRED = 10e-5
TEST_SIMULATION_STEP_SIZE = 0.0001

def check_close(result: np.ndarray, expected: np.ndarray) -> bool:
    if not np.allclose(result, expected, atol=NUMERICAL_ABS_ACCURACY_REQUIRED):
        print(f"expected:{expected}, result:{result}")
        return False
    else:
        return True


def high_accuracy_forward_euler_step(pos: np.ndarray,
                                     vel: np.ndarray,
                                     acc: np.ndarray,
                                     step_size=0.001,
                                     duration=1) -> Tuple[np.ndarray]:
    """
    Update velocity and position for [duration] time,
    using simple Forward-Euler integration rules and Newtonian mechanics.
    Assumes that the acceleration is constant. 

    A simple fool-proof but inaccurate method. Yet with a tiny step size,
    the approximation should be accurate (abeit slow to compute).

    We have the following initival value problem for the position:
    pos' = acc*t
    pos(0) = 0
    """
    vel = vel.astype("float64")
    t = 0
    while t < duration:
        vel += step_size * acc
        pos = pos + step_size * (acc * t)
        t += step_size
    return pos, vel


def runge_kutta_4_step(pos: np.ndarray,
                       vel: np.ndarray,
                       acc: np.ndarray,
                       step_size=0.001,
                       duration=1) -> Tuple[np.ndarray]:
    """
    High order of accuracy approximation of new position and velocity
    after [duration] of time, given constant acceleration.
    """
    for time_step in np.arange(0, duration, step_size):
        k1_v = acc * step_size
        k1_x = vel * step_size

        k2_v = acc * step_size
        k2_x = step_size * (vel + 0.5*k1_v)

        k3_v = acc * step_size
        k3_x = step_size * (vel + 0.5*k2_v)

        k4_v = acc * step_size
        k4_x = step_size * (vel + k3_v)

        pos = pos + (1/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        vel = vel + (1/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return pos, vel