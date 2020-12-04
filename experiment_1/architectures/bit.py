"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

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

Author: Lulof Pirée
October 2020

Architecture implementing the simulation of a single read+write bit register.
"""
import numpy as np
from typing import Tuple, Optional, Dict

from experiment_1.marble_eater_node import MarbleEaterNode
from experiment_1.node import Node, Marble
from experiment_1.attraction_functions.attraction_functions import ThresholdGravity
from experiment_1.architectures.run_and_visualize import run
from experiment_1.auxliary import generate_stiffness_dict
from experiment_1.output_reader import NumMarblesOutputReader
ZERO = np.array([0, 0])

THRESHOLD = 100
ATTRACTION_FUNCTION = ThresholdGravity(THRESHOLD)
NODE_STIFFNESSES = generate_stiffness_dict(marble_stiffness=1,
                                           node_stiffness=1,
                                           marble_attraction=1,
                                           node_attraction=0)
BIT_MARBLE_STIFFNESS = generate_stiffness_dict(marble_stiffness=1,
                                               node_stiffness=0,
                                               marble_attraction=1,
                                               node_attraction=0)
READER_MARBLE_STIFFNESS = generate_stiffness_dict(marble_stiffness=0,
                                                  node_stiffness=1,
                                                  marble_attraction=0,
                                                  node_attraction=0)

LOCKER_NODE_MASS = 50000
LOCKER_POSITIONS = (
    (110, 50),
    (110, 150),
    (23.3974596, 100)
)
BIT_POS = sum(np.array(pos) for pos in LOCKER_POSITIONS)/3


def bit_read_experiment(bit_state: bool = True):
    """
    A bit in state '1' will recoil a reader-Marble into a different angle
    than it approached (because it is just moving towards a position 
    slightly off the center of the bit), and it would pass in a straight line
    if the bit is set to '0'.
    """

    locker_nodes = __generate_locker_nodes(LOCKER_POSITIONS)

    bit_marble = Marble(BIT_POS,
                        ZERO,
                        ZERO,
                        -100,
                        ATTRACTION_FUNCTION,
                        None,
                        **BIT_MARBLE_STIFFNESS)
    reader_marble = __at_angle_toward_bit_marble(BIT_POS,
                                                 0.25*np.pi,
                                                 100,
                                                 READER_MARBLE_STIFFNESS,
                                                 0.03)

    # Any position in line of the movement of the reader Marble,
    # will work as long as it is far enough from the bit not to capture
    # the Marble *before* it hit the bit.
    false_output_reader_pos = reader_marble.pos + 17*reader_marble.vel
    false_output_eater = MarbleEaterNode(false_output_reader_pos,
                                         ZERO,
                                         ZERO,
                                         1,
                                         ATTRACTION_FUNCTION,
                                         radius=6,
                                         **NODE_STIFFNESSES)
    true_output_eater = MarbleEaterNode(np.array([250, 100]),
                                         ZERO,
                                         ZERO,
                                         1,
                                         ATTRACTION_FUNCTION,
                                         radius=6,
                                         **NODE_STIFFNESSES)
    if bit_state:
        marbles = (bit_marble, reader_marble)
    else:
        marbles = (reader_marble,)
    output = run(marbles, locker_nodes + [false_output_eater, true_output_eater])
    print("Read state of bit:", __model_output_to_bit_state(output))


def __generate_locker_nodes(positions: Tuple[Tuple[int]]) -> Node:
    nodes = []
    for pos in positions:
        new_node = Node(pos=np.array(pos),
                        vel=ZERO,
                        acc=ZERO,
                        mass=LOCKER_NODE_MASS,
                        attraction_function=ATTRACTION_FUNCTION,
                        **NODE_STIFFNESSES)
        nodes.append(new_node)
    return nodes


def __heads_on_to_bit_marble(locker_positions) -> Marble:
    """
    Create a Marble that horizontally collides into the bit center,
    approaching from the left.
    """
    pos = np.array([
        max(map(lambda x: x[0], locker_positions)),
        sum(map(lambda x: x[1], locker_positions))/len(locker_positions)
    ])
    reader_marble = Marble(pos,
                           np.array([-10, 0]),
                           ZERO,
                           100,
                           ATTRACTION_FUNCTION,
                           None,
                           **READER_MARBLE_STIFFNESS)
    return reader_marble


def __at_angle_toward_bit_marble(bit_position: np.ndarray,
                                 angle: float,
                                 horizontal_distance: float,
                                 stiffnesses: Dict[str, float],
                                 x_error: Optional[float] = 0,
                                 mass: Optional[float] = 100,
                                 speed_multiplier: Optional[float] = 0.1
                                 ) -> Marble:
    """
    Create a Marble that collides into the bit center,
    approaching from the top-left at the given angle.
    """
    vertical_distance = np.tan(angle)*horizontal_distance
    pos = np.array([bit_position[0] + horizontal_distance + x_error,
                    bit_position[1] + vertical_distance])
    reader_marble = Marble(pos,
                           speed_multiplier*(bit_position - pos) +
                           np.array([x_error, 0]),
                           ZERO,
                           mass,
                           ATTRACTION_FUNCTION,
                           None,
                           **stiffnesses)
    return reader_marble


def __model_output_to_bit_state(model_output: np.array) -> bool:
    """
    Translate the output of a NenwinModel simulating the reading of
    a Boolean bit register to the Boolean state read.
    """
    if np.isclose(model_output[0], 1):
        return False
    elif np.isclose(model_output[1], 1):
        return True
    else:
        raise RuntimeError("model_output could not be interpreted")


def bit_write_experiment():
    """
    Propelling a Marble into a '0' bit,
    and ensuring it becomes fixed in the center of the bit -> '1' bit.

    Note: this seems not to work when the distance is too great.
    Then the speed needs to be fine-tunes more accurately than
    numerical precision allows. 
    """
    locker_nodes = __generate_locker_nodes(LOCKER_POSITIONS)

    writer_marble1 = __at_angle_toward_bit_marble(BIT_POS,
                                                  0.0992*np.pi,
                                                  100,
                                                  BIT_MARBLE_STIFFNESS,
                                                  speed_multiplier=0.5,
                                                  mass=-100)
    writer_marble2 = __at_angle_toward_bit_marble(BIT_POS,
                                                  0,
                                                  100,
                                                  BIT_MARBLE_STIFFNESS,
                                                  speed_multiplier=0.439002025,
                                                  mass=-100)
    writer_marble3 = __at_angle_toward_bit_marble(BIT_POS,
                                                  0*np.pi,
                                                  1,
                                                  BIT_MARBLE_STIFFNESS,
                                                  speed_multiplier=-0.1,
                                                  mass=-100)
    run([writer_marble1, writer_marble2, writer_marble3], locker_nodes)


if __name__ == "__main__":
    bit_read_experiment(True)
    # bit_write_experiment()
