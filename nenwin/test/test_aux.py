"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Copyright (C) 2020 Lulof Pirée, Teun Schilperoort

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Settings (constants) for the test cases, such as required numerical precision,
and auxiliary functions.
"""
import numpy as np
import torch
from typing import Tuple, Dict

from nenwin.attraction_functions.attraction_functions \
    import ConstantAttraction
from nenwin.particle import PhysicalParticle
NUMERICAL_ABS_ACCURACY_REQUIRED = 10e-5
TEST_SIMULATION_STEP_SIZE = 0.001
ZERO = torch.tensor([0], dtype=torch.float)

ATTRACT_FUNCT = ConstantAttraction()


class MockPipe:

    def poll(self):
        return None


def check_close(result: torch.Tensor,
                expected: torch.Tensor,
                atol=NUMERICAL_ABS_ACCURACY_REQUIRED
                ) -> bool:
    if not torch.allclose(result, expected, atol=atol):
        print(f"expected:{expected}, result:{result}")
        return False
    else:
        return True

def check_named_parameters(expected: Dict[str, object],
                       named_parameters: Tuple[Tuple[str, torch.Tensor]]
                       ) -> bool:
    """
    Given a tuple of named parameters 
    (as returned by torch.nn.Module instances),
    computes if *at least* all the named with corresponding values of the
    named_parameters dict occur.
    """
    output = True
    for name, param in named_parameters:
            if name in set(expected.keys()):
                expected_value = expected.pop(name)
                if not isinstance(expected_value, torch.Tensor):
                    expected_value = torch.Tensor([expected_value])
                output = output and check_close(expected_value, param)
    output = output and (len(expected) == 0)

    if not output:
        print(f"check_named_parameters: Remaining names: {expected}")
    return output


def high_accuracy_forward_euler_step(pos: torch.Tensor,
                                     vel: torch.Tensor,
                                     acc: torch.Tensor,
                                     step_size=0.001,
                                     duration=1) -> Tuple[torch.Tensor]:
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

def convert_scalar_param_to_repr(scalar: float) -> str:
    """
    When converting a scalar to a single-element torch.Tensor,
    the last digits may change a little because of conversion.
    Hence the repr() of the torch.Tensor version is a little
    different than the repr() of the float itself.
    """
    return repr(torch.tensor(scalar, dtype=torch.float).item())


def runge_kutta_4_step(pos: torch.Tensor,
                       vel: torch.Tensor,
                       acc: torch.Tensor,
                       step_size=0.001,
                       duration=1) -> Tuple[torch.Tensor]:
    """
    High order of accuracy approximation of new position and velocity
    after [duration] of time, given constant acceleration.
    """
    pos = pos.clone()
    vel = vel.clone()

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
