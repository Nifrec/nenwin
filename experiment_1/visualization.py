"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Simple graphical visualization of a running simulation.
"""
import numpy as np
from typing import Tuple
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

from experiment_1.model import NenwinModel



NODE_COLOUR = pygame.Color(255, 143, 95)
MARBLE_COLOUR = pygame.Color(155, 255, 95)
BACKGROUND_COLOR = pygame.Color(225, 201, 255)
RADIUS_SCALE_FACTOR = 0.1


class NenwinVisualization():
    """
    Creates and updates a visualization of a model.

    Note that is a simple, naive low-performance implementation.
    """

    def __init__(self, resolution: Tuple[int], model: NenwinModel):
        """
        Arguments:
        * resolution: (width, height) of window. Note that the simulation
            is not scaled to fit in the resolution, so not all particles
            might be visible for a small resolution.
        * model: the NenwinModel to visualize. The model should not be running.
        """
        pygame.init()
        pygame.display.set_caption("Nenwin")
        self.__resolution = resolution
        self.__model = model
        self.__screen = pygame.display.set_mode((resolution[0], resolution[1]))

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
            pygame.draw.circle(surf,
                               NODE_COLOUR,
                               np.round(node.pos).astype(int),
                               round(node.mass * RADIUS_SCALE_FACTOR))
        for marble in self.__model.marbles:
            pygame.draw.circle(surf,
                               MARBLE_COLOUR,
                               np.round(marble.pos).astype(int),
                               round(marble.mass * RADIUS_SCALE_FACTOR))
        
        return surf

    def __process_events(self):
        """
        Registers button- and keypresses and executes corresponding effect.
        """
        # Now process pygame events (keyboard keys or controller buttons)
        for event in pygame.event.get():

            if (event.type == pygame.QUIT): # If user closes pygame window
                self.__is_running = False
            
            elif (event.type == pygame.KEYDOWN): # If a button is pressed
                if (event.key == pygame.K_ESCAPE): # User pressed ESC
                    self.__is_running = False

    def run(self, steps_per_display_update: int):
        """
        Run the model and update the visualization every 
        [steps_per_display_update] steps of the simulation.
        """
        self.__is_running = True

        while self.__is_running:
            self.__draw_all_particles()
            self.__model.run(max_num_steps=steps_per_display_update)
            self.__process_events()
