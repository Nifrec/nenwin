"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Simple graphical visualization of a running simulation.
"""
from os import environ
# Disable pygame welcome message. Need to be set before 'import pygame'
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from typing import Tuple
from experiment_1.model import NenwinModel


class NenwinVisualization():
    """
    Creates and updates a visualization of a model.
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

    def run(self, steps_per_display_update: int):
        """
        Run the model and update the visualization every 
        [steps_per_display_update] steps of the simulation.
        """
        #TODO: draw stuff

        self.__model.run(max_num_steps=steps_per_display_update)
