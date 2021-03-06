"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Teun Schilperoort
Feb 2021

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

Class to place a set of input values into a given space.
"""
import abc
import numpy as np
from typing import Iterable
import math

from nenwin.node import Marble

class InputPlacer(abc.ABC):
    """
    Abstract class providing the interface for classes
    that assign a set of input values a position in the network.
    """

    def __init__(self, input_pos: np.array, input_region_sizes: np.array):
        """
        Arguments:

        * input_pos: vertex of input region with lowest distance to the origin
            (lower left corner in 2D, generalizes to higher dimensions)
        * input_region_sizes: array of lengths of input region per dimension
            (in 2D, this are the width and the height respectively)
        """
        self.__input_pos = input_pos
        self.__input_region_sizes = input_region_sizes

    @property
    def input_pos(self):
        return self.__input_pos

    @property
    def input_region_sizes(self):
        return self.__input_region_sizes

    @abc.abstractmethod
    def marblize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        """
        Given a set of data, create a Marble for each input datum,
        and assign them a position in the input region.
        """
        pass

class PhiInputPlacer(InputPlacer):
    """
    Given a set of data, create the marbles and place them from the bottomleft
    to topright according to a pseudorandom sequence
    """

    def marblize_data(self, input_data: Iterable[object]) -> Iterable[Marble]:
        list_phi = [0, 1.618033988749895, 1.324717957244746, 1.2207440846057596, 
                    1.1673039782614187, 1.1347241384015194, 1.1127756842787055, 
                    1.0969815577985598, 1.085070245491451, 1.0757660660868371, 
                    1.0682971889208412, 1.062169167864255, 1.0570505752212285, 
                    1.0527109201475582, 1.0489849347570344, 1.0457510241563417, 
                    1.0429177323017866, 1.0404149477818474, 1.03818801943645, 
                    1.0361937171306834, 1.034397396133807]
        
        print(input_data)
        input_list = list(input_data)
        dimension = len(input_list[0])
        dimension_vector = np.array([])
        for i in range(1, dimension + 1):
            dimension_vector = np.append(dimension_vector, np.divide(1, list_phi[dimension]**i))
        #print("dimension_vector", dimension_vector)
        
        number_of_data = len(input_list)
        
        exists = False
        for i in range(0, number_of_data):
            if not exists:
                possible_points = [divmod(dimension_vector,1)[1]]
                exists = True
            else:    
                possible_points = np.append(possible_points, [divmod((i+1)*dimension_vector, 1)[1]], axis = 0)
        print("possible points", list(possible_points))
        
        """
        Works, step 1 and 2 for the algorithm in latex, possible_point is array of points op to i
        """
        
        #find point closest to bottom_left point = input_pos, is equal to entry of possible 
        #points with shortest length
        length_possible_points = []
        for entry in possible_points:
            length = 0
            for dimension in entry:
                length = length + dimension**2  
            length_possible_points.append(math.sqrt(length))
        print("lengths", length_possible_points)
        
        minimal_point_index = np.where(length_possible_points == np.amin(length_possible_points))[0]
        distance_sequence = np.array([possible_points[minimal_point_index]])
        print("distance_sequence", distance_sequence)
        
        #distance sequence is filled with bottom-left most point
        #below is incorrect, fix next time
        #testcase: PhiInputPlacer.marblize_data(a, input_data =[[1,2,67,9,6],[1,3,4,5,6],[3,3,3,3,3]])
        possible_points_edit = possible_points
        for i in range(0, number_of_data):    
            current_point = possible_points[minimal_point_index]
            possible_points_edit = np.delete(possible_points_edit, possible_points_edit.argmin())
            print(list(possible_points_edit))
            list_distances = []
            for entry in possible_points_edit:
                distance = 0
                for i in range(0, int(dimension)):
                    print("k",current_point)
                    distance = (current_point[i] - entry[i])**2
                list_distances.append(distance)
                #print(list_distances)
        
        pass