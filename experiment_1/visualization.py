"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Simple graphical visualization of a running simulation.
"""
from typing import Tuple, Optional 
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
from numbers import Number

from experiment_1.model import NenwinModel
from experiment_1.simulation import Simulation
from experiment_1.stiffness_particle import Node
from experiment_1.marble_eater_node import MarbleEaterNode




NODE_COLOUR = pygame.Color(255, 143, 95)
MARBLE_COLOUR = pygame.Color(155, 255, 95)
BACKGROUND_COLOR = pygame.Color(10, 10, 10)
NODE_RADIUS = 10
MARBLE_RADIUS = 5


class NenwinVisualization():
    """
    Creates and updates a visualization of a model.

    Note that is a simple, naive low-performance implementation.
    """

    def __init__(self,
                 resolution: Tuple[int],
                 simulation: Simulation,
                 model: NenwinModel,
                 scale_factor: int = 1):
        """
        Arguments:
        * resolution: (width, height) of window. Note that the simulation
            is not scaled to fit in the resolution, so not all particles
            might be visible for a small resolution.
        * model: the NenwinModel to visualize. The model should not be running.
        * scale_factor: factor for scaling coordinates of objects (by default
            position directly correspond to pixel coordinates)
        """
        pygame.init()
        pygame.display.set_caption("Nenwin")
        self.__resolution = resolution
        self.__simulation = simulation
        self.__model = model
        self.__screen = pygame.display.set_mode((resolution[0], resolution[1]))
        self.__scale_factor = scale_factor

    def __draw_all_particles(self):
        new_frame = self.__create_new_frame()
        self.__screen.blit(new_frame, (0, 0))
        pygame.display.flip()

    def __create_new_frame(self) -> pygame.Surface:
        """
        Draw all particles to an opaque Surface with the same
        resolution as the window.
        """
        surf = pygame.Surface(self.__resolution)
        surf.fill(BACKGROUND_COLOR)
        for node in self.__model.nodes:
            radius = self.__find_radius_of_node(node)
            pygame.draw.circle(surf,
                               NODE_COLOUR,
                               np.round(self.__scale_factor *
                                        node.pos).astype(int),
                               int(np.round(self.__scale_factor*radius).flatten()))
        for marble in self.__model.marbles:
            pygame.draw.circle(surf,
                               MARBLE_COLOUR,
                               np.round(self.__scale_factor *
                                        marble.pos).astype(int),
                               MARBLE_RADIUS)

        return surf

    def __find_radius_of_node(self, node: Node) -> float:
        """
        Return default radius of nodes if an ordinairy node
        (without any 'radius' attribute),
        or use the specified radius in case of an Eater.
        """
        if isinstance(node, MarbleEaterNode):
            return node.radius
        else:
            return NODE_RADIUS

    def __process_events(self):
        """
        Registers button- and keypresses and executes corresponding effect.
        """
        # Now process pygame events (keyboard keys or controller buttons)
        for event in pygame.event.get():

            if (event.type == pygame.QUIT):  # If user closes pygame window
                self.__is_running = False

            elif (event.type == pygame.KEYDOWN):  # If a button is pressed
                if (event.key == pygame.K_ESCAPE):  # User pressed ESC
                    self.__is_running = False

    def run(self,
            simulation_steps_per_frame: int,
            step_size: Number,
            max_num_frames: Optional[int] = None):
        """
        Run the model and update the visualization every 
        [simulation_steps_per_frame] steps of the simulation.
        """
        self.__is_running = True
        frame_number = 0

        while self.__is_running:
            self.__draw_all_particles()
            self.__simulation.run(step_size=step_size,
                                  max_num_steps=simulation_steps_per_frame)
            self.__process_events()

            frame_number += 1
            if max_num_frames is not None and frame_number >= max_num_frames:
                self.__is_running = False
