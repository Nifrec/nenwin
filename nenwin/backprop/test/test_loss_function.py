"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

Copyright (C) 2021 Lulof Pirée, Teun Schilperoort

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

Testcases for the source file loss_function.py
"""
from typing import List, Sequence
import torch
import unittest

from nenwin.marble_eater_node import MarbleEaterNode
from nenwin.node import Marble, Node
from nenwin.model import NenwinModel
from nenwin.auxliary import distance
from nenwin.backprop.loss_function import LossCases, NenwinLossFunction, find_closest_marble_to, find_most_promising_marble_to, \
    velocity_weighted_distance
from nenwin.attraction_functions.attraction_functions import NewtonianGravity

ZERO = torch.tensor([0.0, 0.0])


class FindClosestMarbleToTestCase(unittest.TestCase):

    def test_find_closest_marble_to_1(self):
        """
        Base case: target particle not in Model.
        """
        marble_positions = (
            (10, 10),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          ZERO, ZERO, 0, None, None)
                   for pos in marble_positions]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([12.87, 2.9]), ZERO, ZERO, 0, None, None)

        expected = marbles[2]

        self.assertIs(find_closest_marble_to(target, model), expected)

    def test_find_closest_marble_to_2(self):
        """
        Base case: target particle in Model.
        """
        marble_positions = (
            (10, 10),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7),
            (11.1, 3.5)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          ZERO, ZERO, 0, None, None)
                   for pos in marble_positions]
        model = NenwinModel([], marbles)

        target = marbles[2]

        expected = marbles[4]

        self.assertIs(find_closest_marble_to(target, model), expected)

    def test_error_if_no_other_marbles(self):
        """
        Cannot find closest Marble if no other Marbles exist.
        """
        target = Marble(torch.tensor([12.87, 2.9]), ZERO, ZERO, 0, None, None)
        model = NenwinModel([], [target])

        with self.assertRaises(RuntimeError):
            find_closest_marble_to(target, model)


class FindMostPromisingMarbleTo(unittest.TestCase):

    def test_find_most_promising_marble_to_1(self):
        """
        Base case: target particle not in Model. Both weights are 1.
        """
        marble_positions = (
            (0, 0),
            (10.2, 10.1),
            (5, 10),
            (12.9, 3.2),
            (9.9, -0.7)
        )

        marble_velocities = (
            (0, 0),
            (5, 5.1),
            (10, -100),
            (1.2, 1),
            (0, 0)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          torch.tensor(vel, dtype=torch.float),
                          ZERO, 0, None, None)
                   for pos, vel in zip(marble_positions, marble_velocities)]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([15.0, 15.0]), ZERO, ZERO, 0, None, None)

        expected = marbles[1]

        self.assertIs(find_most_promising_marble_to(
            target, model, 1, 1), expected)

    def test_find_most_promising_marble_to_2(self):
        """
        Base case: target particle not in Model. Weights differ.
        """
        marble_positions = (
            (0, 0),
            (10.0, 10.0)
        )

        marble_velocities = (
            (7.5, 7.5),
            (5.0, 5.0)
        )

        marbles = [Marble(torch.tensor(pos, dtype=torch.float),
                          torch.tensor(vel, dtype=torch.float),
                          ZERO, 0, None, None)
                   for pos, vel in zip(marble_positions, marble_velocities)]
        model = NenwinModel([], marbles)

        target = Marble(torch.tensor([15.0, 15.0]), ZERO, ZERO, 0, None, None)

        expected = marbles[0]

        pos_weight = 1
        vel_weight = 2
        result = find_most_promising_marble_to(target, model,
                                               pos_weight, vel_weight)
        self.assertIs(result, expected)


class VelocityWeightedDistanceTestCase(unittest.TestCase):

    def test_velocity_weighted_distance_1(self):
        """
        Base case: 0 distance, both weights 1.
        """
        pos_1 = torch.tensor([15.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([15.0, 15.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(0.0)
        result = velocity_weighted_distance(m1, m2, 1, 1)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_2(self):
        """
        Base case: vel weight not zero.
        """
        pos_1 = torch.tensor([0.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([0.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(25.0)
        pos_weight = 1
        vel_weight = 1/3
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_3(self):
        """
        Corner case: pos weight zero.
        """
        pos_1 = torch.tensor([0.0, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([0.0, 0.0])
        vel_2 = torch.tensor([0.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = torch.tensor(100.0)
        pos_weight = 0
        vel_weight = 1/3
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)

    def test_velocity_weighted_distance_3(self):
        """
        Corner case: vel weight zero.
        """
        pos_1 = torch.tensor([123, 15.0])
        m1 = Marble(pos_1, ZERO, ZERO, 0, None, None)

        pos_2 = torch.tensor([4.3, 2.4])
        vel_2 = torch.tensor([89.0, 30.0])
        m2 = Marble(pos_2, vel_2, ZERO, 0, None, None)

        expected = distance(m1, m2)**2
        pos_weight = 1
        vel_weight = 0
        result = velocity_weighted_distance(m1, m2, pos_weight, vel_weight)

        torch.testing.assert_allclose(result, expected)


class FindLossCaseTestCase(unittest.TestCase):

    def test_no_output(self):
        nodes = [gen_node_at(torch.tensor([10., 10]))]
        marbles = [gen_marble_at(ZERO)]
        model = NenwinModel(nodes, marbles)
        loss_fun = NenwinLossFunction(nodes, model, 0, 1)

        result = loss_fun._find_loss_case(0)
        expected = LossCases.no_prediction

        self.assertEqual(result, expected)

    def test_wrong_output(self):
        nodes = [gen_node_at(torch.tensor([10., 10])),
                 gen_node_at(torch.tensor([11., 11]))]
        marbles = [gen_marble_at(ZERO), gen_marble_at(torch.tensor([11., 11]))]
        model = NenwinModel(nodes, marbles)
        nodes[1].eat(marbles[1])
        loss_fun = NenwinLossFunction(nodes, model, 0, 1)

        target_index = 0  # Node at index 1 ate the Marble
        result = loss_fun._find_loss_case(target_index)
        expected = LossCases.wrong_prediction

        self.assertEqual(result, expected)

    def test_correct_output(self):
        node = gen_node_at(ZERO)
        marble = gen_marble_at(ZERO)
        model = NenwinModel([node], [marble])
        node.eat(marble)
        loss_fun = NenwinLossFunction([node], model, 0, 1)

        target_index = 0
        result = loss_fun._find_loss_case(target_index)
        expected = LossCases.correct_prediction

        self.assertEqual(result, expected)


class LossFunctionCallCorrectPredTestCase(unittest.TestCase):
    """
    Testcases for NenwinLossFunction.__call__() in case the output is correct.
    """

    def test_value_loss_correct_output(self):
        """
        If the prediction is correct, the loss should be 0.0.
        """
        node = gen_node_at(ZERO)
        marble = gen_marble_at(ZERO)
        model = NenwinModel([node], [marble])
        node.eat(marble)
        loss_fun = NenwinLossFunction([node], model, 0, 1)

        target_index = 0
        result = loss_fun(target_index)
        expected = 0.0

        self.assertAlmostEqual(result, expected)

    def test_grads_no_error_correct_pred(self):
        """
        Case in which the prediction is correct.
        All learnable parameters should have a gradient value
        that does not cause errors with an optimizer.

        Passes if no errors occurs.
        """
        node = gen_node_at(ZERO)
        marble = gen_marble_at(ZERO)
        model = NenwinModel([node], [marble])
        model.make_timestep(1.0)
        optim = torch.optim.Adam(model.parameters())
        optim.zero_grad()

        loss_fun = NenwinLossFunction([node], model, 0, 1)

        target_index = 0
        result = loss_fun(target_index)

        try:
            result.backward()
            optim.step()
        except RuntimeError as e:
            self.fail(f"Error occurred during backprop/optim step: {e}")

    def test_grads_value_correct_pred(self):
        """
        Case in which the prediction is correct.
        All learnable parameters should have a gradient value
        that does not affect the values of the weights.
        """
        node = gen_node_at(ZERO)
        marble = gen_marble_at(torch.tensor([1.1, 1.1]))
        model = NenwinModel([node], [marble])
        for _ in range(1000):
            model.make_timestep(0.1)

        assert node.num_marbles_eaten == 1, "Testcase badly desgined."

        loss_fun = NenwinLossFunction([node], model, 0, 1)

        self.target_index = 0  # Node at index 1 ate the Marble
        result = loss_fun(self.target_index)
        result.backward(retain_graph=True)

        self.assertIsNone(marble.init_pos.grad)
        self.assertIsNone(marble.init_vel.grad)
        self.assertIsNone(marble.mass.grad)


class LossFunctionCallNoPredTestCase(unittest.TestCase):
    """
    Testcases for NenwinLossFunction.__call__() in case none of the
    output-MarbleEaterNodes has eaten any Marble.
    """

    def setUp(self):

        self.pos_weight = 0.5
        self.vel_weight = 0.5

        self.node = gen_node_at(ZERO)
        self.marble = gen_marble_at(torch.tensor([10., 10.]))
        self.model = NenwinModel([self.node], [self.marble])
        self.model.make_timestep(0.1)  # too short to cross the distance

        self.loss_fun = NenwinLossFunction([self.node], self.model,
                                           vel_weight=self.vel_weight,
                                           pos_weight=self.pos_weight)

        assert self.node.num_marbles_eaten == 0, "Testcase badly desgined."

        self.target_index = 0
        self.loss = self.loss_fun(self.target_index)

    def test_value_loss_no_output(self):
        """
        If no Marble has been eaten, the loss
        should equal the velocity-weighted distance
        of the target Node to the nearest Marble.
        """
        expected = velocity_weighted_distance(self.node, self.marble,
                                              pos_weight=self.pos_weight,
                                              vel_weight=self.vel_weight)
        torch.testing.assert_allclose(self.loss, expected)

    def test_grads_no_error_no_output(self):
        """
        Case in which no prediction output is given.
        All learnable parameters should have a gradient value
        that does not cause errors with an optimizer.

        Passes if no errors occurs.
        """
        optim = torch.optim.Adam(self.model.parameters())
        optim.zero_grad()

        try:
            self.loss.backward()
            optim.step()
        except RuntimeError as e:
            self.fail(f"Error occurred during backprop/optim step: {e}")

    def test_grads_value_no_output(self):
        """
        Case in which no prediction output is given.
        The learnable parameters of the Marble should have a gradient value
        that does affect the values of the weights.
        The loss should be lower for the second run.
        """
        optim = torch.optim.Adam(self.model.parameters())
        optim.zero_grad()

        self.loss.backward()

        self.assertIsNotNone(self.marble.init_pos.grad)
        self.assertIsNotNone(self.marble.init_vel.grad)
        self.assertIsNotNone(self.marble.mass.grad)

        optim.step()

        # Now verify the loss improved.
        self.model.reset()
        self.model.make_timestep(0.1)
        new_loss = self.loss_fun(self.target_index)

        self.assertLess(new_loss.item(), self.loss.item())


class LossFunctionCallWrongPredTestCase(unittest.TestCase):
    """
    Testcases for NenwinLossFunction.__call__() in another than the target
    output-MarbleEaterNodes has eaten a Marble.
    """

    def setUp(self):
        """

        Sketch:

         |
        ^|
        y|
         |
        0| N_1        <M        N_0
         |
         |
         +-------------------------------
           -10         0         10   x>
        

        Marble M starts with moving towards N_1, but should arrive at N_0.

        """
        self.pos_weight = 0.5
        self.vel_weight = 0.5

        self.nodes = (gen_node_at(torch.tensor([10.0, 0])),
                      gen_node_at(torch.tensor([-10.0, 0])))
        self.marble = gen_marble_at(torch.tensor([0.0]),
                                    vel=torch.tensor([-3.0, 0]))
        self.model = NenwinModel([self.node], [self.marble])

        for _ in range(50): # Should be enough for the Marble to be eaten
            self.model.make_timestep(0.1)  

        assert len(self.model.marbles), "Testcase badly desgined."

        self.loss_fun = NenwinLossFunction([self.node], self.model,
                                           vel_weight=self.vel_weight,
                                           pos_weight=self.pos_weight)

        self.target_index = 0
        self.loss = self.loss_fun(self.target_index)

    def test_value_loss_wrong_pred_no_marble_left(self):
        """
        The loss should equal the velocity-weighted distance
        of the target Node to the nearest Marble,
        plus the *negative reciprocal* of the distance of the wrong node
        (self.nodes[1]) to the Marble.

        Case where no non-eaten Marble is available.
        """
        self.fail()
        expected = velocity_weighted_distance(self.node, self.marble,
                                              pos_weight=self.pos_weight,
                                              vel_weight=self.vel_weight)
        torch.testing.assert_allclose(self.loss, expected)

    def test_value_loss_wrong_pred_some_marble_left(self):
        """
        The loss should equal the velocity-weighted distance
        of the target Node to the nearest Marble,
        plus the *negative reciprocal* of the distance of the wrong node
        (self.nodes[1]) to the Marble.

        Case where another non-eaten Marble is still available
        at time of loss computation.
        """
        self.fail()
        expected = velocity_weighted_distance(self.node, self.marble,
                                              pos_weight=self.pos_weight,
                                              vel_weight=self.vel_weight)
        torch.testing.assert_allclose(self.loss, expected)

    def test_grads_no_error_wrong_pred(self):
        self.fail()
        """
        Case in which no prediction output is given.
        All learnable parameters should have a gradient value
        that does not cause errors with an optimizer.

        Passes if no errors occurs.
        """
        optim = torch.optim.Adam(self.model.parameters())
        optim.zero_grad()

        try:
            self.loss.backward()
            optim.step()
        except RuntimeError as e:
            self.fail(f"Error occurred during backprop/optim step: {e}")

    def test_grads_value_wrong_pred(self):
        self.fail()
        """
        Case in which no prediction output is given.
        The learnable parameters of the Marble should have a gradient value
        that does affect the values of the weights.
        The loss should be lower for the second run.
        """
        optim = torch.optim.Adam(self.model.parameters())
        optim.zero_grad()

        self.loss.backward()

        self.assertIsNotNone(self.marble.init_pos.grad)
        self.assertIsNotNone(self.marble.init_vel.grad)
        self.assertIsNotNone(self.marble.mass.grad)

        optim.step()

        # Now verify the loss improved.
        self.model.reset()
        self.model.make_timestep(0.1)
        new_loss = self.loss_fun(self.target_index)

        self.assertLess(new_loss.item(), self.loss.item())


def gen_node_at(pos: torch.Tensor, mass: float = 10) -> MarbleEaterNode:
    output = MarbleEaterNode(pos, vel=ZERO, acc=ZERO,
                             mass=mass,
                             attraction_function=NewtonianGravity(),
                             marble_attraction=1,
                             marble_stiffness=1,
                             node_attraction=0,
                             node_stiffness=0,
                             radius=1)

    return output


def gen_marble_at(pos: torch.Tensor,
                  vel: torch.Tensor = ZERO,
                  mass: float = 10) -> Marble:
    output = Marble(pos=pos, vel=vel, acc=ZERO,
                    mass=mass,
                    attraction_function=NewtonianGravity(),
                    marble_attraction=1,
                    marble_stiffness=1,
                    node_attraction=0,
                    node_stiffness=0,
                    datum=None)

    return output


if __name__ == '__main__':
    unittest.main()
