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
import torch
from typing import Tuple, Optional, Dict

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Node, Marble
from nenwin.attraction_functions.attraction_functions \
    import ThresholdGravity, ConstantAttraction
from nenwin.architectures.run_and_visualize import run
from nenwin.auxliary import generate_stiffness_dict, generate_node_dict,\
    distance
from nenwin.output_reader import NumMarblesOutputReader
from nenwin.marble_emitter_node import MarbleEmitterNode, MarbleEmitter
ZERO = np.array([0, 0])

THRESHOLD = 100
ATTRACTION_FUNCTION = ThresholdGravity(THRESHOLD)
ZERO_ATTRACTION = ConstantAttraction(0)
NODE_STIFFNESSES = generate_stiffness_dict(marble_stiffness=1,
                                           node_stiffness=1,
                                           marble_attraction=1,
                                           node_attraction=0)
EMITTER_STIFFNESSES = generate_stiffness_dict(marble_stiffness=1,
                                              node_stiffness=1,
                                              marble_attraction=0,
                                              node_attraction=0)
BIT_MARBLE_STIFFNESS = generate_stiffness_dict(marble_stiffness=1,
                                               node_stiffness=0,
                                               marble_attraction=1,
                                               node_attraction=0)
READER_MARBLE_STIFFNESS = generate_stiffness_dict(marble_stiffness=0,
                                                  node_stiffness=1,
                                                  marble_attraction=0,
                                                  node_attraction=0)

LOCKER_NODE_MASS = 100000
LOCKER_POSITIONS = (
    (110, 50),
    (110, 150),
    (23.3974596, 100)
)
BIT_MARBLE_MASS = -100
BIT_POS = sum(np.array(pos) for pos in LOCKER_POSITIONS)/3
BIT_MARBLE_SETTTINGS = generate_node_dict(BIT_POS,
                                          ZERO,
                                          ZERO,
                                          BIT_MARBLE_MASS,
                                          ATTRACTION_FUNCTION,
                                          **BIT_MARBLE_STIFFNESS)
BIT_MARBLE_SETTTINGS["datum"] = None


def bit_read_experiment(bit_state: bool = True):
    """
    A bit in state '1' will recoil a reader-Marble into a different angle
    than it approached (because it is just moving towards a position
    slightly off the center of the bit), and it would pass in a straight line
    if the bit is set to '0'.
    """

    locker_nodes = __generate_locker_nodes(LOCKER_POSITIONS)

    reader_marble = __at_angle_toward_bit_marble(BIT_POS,
                                                 0.25*np.pi,
                                                 100,
                                                 READER_MARBLE_STIFFNESS,
                                                 0.03)
    bit_marble = Marble(**BIT_MARBLE_SETTTINGS)

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
    output = run(marbles, locker_nodes +
                 [false_output_eater, true_output_eater])
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


def __at_angle_toward_bit_marble(bit_position: torch.Tensor,
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


def bit_write_experiment_propelling():
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
                                                  mass=BIT_MARBLE_MASS)
    writer_marble2 = __at_angle_toward_bit_marble(BIT_POS,
                                                  0,
                                                  100,
                                                  BIT_MARBLE_STIFFNESS,
                                                  speed_multiplier=0.439002025,
                                                  mass=BIT_MARBLE_MASS)
    writer_marble3 = __at_angle_toward_bit_marble(BIT_POS,
                                                  0*np.pi,
                                                  1,
                                                  BIT_MARBLE_STIFFNESS,
                                                  speed_multiplier=-0.1,
                                                  mass=BIT_MARBLE_MASS)
    run([writer_marble1, writer_marble2, writer_marble3], locker_nodes)


def bit_write_expetiment_emitters():
    locker_nodes = __generate_locker_nodes(LOCKER_POSITIONS)

    # Unity vector from lower-left-corner towards upper-right-corner
    incoming_direction = 0.5*np.sqrt(2) * np.array([1, -1])
    outer_emitter_pos = BIT_POS - 50*incoming_direction
    inner_emitter_pos = BIT_POS - 0.1*incoming_direction
    outer_radius = 5

    signal_marble_pos = outer_emitter_pos + outer_radius*(incoming_direction)
    signal_marble_mass = BIT_MARBLE_MASS
    signal_marble_prototype = Marble(pos=signal_marble_pos,
                                     vel=10*incoming_direction,
                                     acc=ZERO,
                                     mass=signal_marble_mass,
                                     attraction_function=ZERO_ATTRACTION,
                                     datum=None,
                                     **READER_MARBLE_STIFFNESS)
    outer_emitter = MarbleEmitter(signal_marble_prototype, 0, 0)
    outer_emitter_node = MarbleEmitterNode(outer_emitter_pos,
                                           ZERO,
                                           ZERO,
                                           1,
                                           ZERO_ATTRACTION,
                                           radius=outer_radius,
                                           emitter=outer_emitter,
                                           **EMITTER_STIFFNESSES)
    inner_emitter_pos = BIT_POS - 2*(incoming_direction)
    inner_emitter_radius = 0.01
    bit_spawn_pos = inner_emitter_pos + \
        incoming_direction*inner_emitter_radius

    bit_marble = Marble(pos=bit_spawn_pos,
                        vel=np.array([1, 0.5]),
                        acc=ZERO,
                        mass=BIT_MARBLE_MASS,
                        attraction_function=ATTRACTION_FUNCTION,
                        datum=None,
                        **BIT_MARBLE_STIFFNESS)
    inner_emitter = MarbleEmitter(bit_marble, 0)
    inner_emitter_node = MarbleEmitterNode(inner_emitter_pos,
                                           ZERO,
                                           ZERO,
                                           1,
                                           ZERO_ATTRACTION,
                                           radius=inner_emitter_radius,
                                           emitter=inner_emitter,
                                           **EMITTER_STIFFNESSES)

    activation_marble = Marble(pos=outer_emitter_pos + np.array([0, 100]),
                               vel=np.array([0, -10]),
                               acc=ZERO,
                               mass=signal_marble_mass,
                               attraction_function=ATTRACTION_FUNCTION,
                               datum=None,
                               **READER_MARBLE_STIFFNESS)

    run([activation_marble], locker_nodes +
        [outer_emitter_node, inner_emitter_node])


if __name__ == "__main__":
    # bit_read_experiment(True)
    # bit_write_experiment_propelling()
    bit_write_expetiment_emitters()


"""
Notes:

Making the Bit nodes too heavy increases any oscilations of the bit marble,
as it experiences greater forces.
Changing the mass of the bit marble itself does not make any difference,
(when using Newtonion gravity):
as      F = m_1 * m_2 / r
and     acc = F / m_1
Hence   acc = m_2 / r
"""
