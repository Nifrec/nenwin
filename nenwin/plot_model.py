"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée

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

--------------------------------------------------------------------------------

Simple function to plot a Nenwin model using Matplotlib.
"""
from __future__ import annotations

import torch
import torch.nn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple

from nenwin.model import NenwinModel


def plot_model(model: NenwinModel, ax: Axes = None
               ) -> Tuple[Figure, Axes] | None:
    """
    Plot the particles of a NenwinModel in a Matplotlib graph.
    Each Node will be rendered as an orange dot,
    and each Marble as a green dot.
    Only supports models where the position of particles is two-dimensional.

    Returns a new Matplotlib Figure and Axes containing the resulting plot.
    Alternatively, an axis can be provided to which the plot will be added.
    """

    partiles = set(model.nodes)
    partiles.update(model.marbles)

    ax_provided = True
    if ax is None:
        ax_provided = False
        fig, ax = plt.subplots(1, 1)

    for node in model.nodes:
        assert torch.numel(node.pos) == 2
        point_coords = node.pos.detach().numpy().reshape((2))
        nodes, = ax.plot(point_coords[0], point_coords[1],
                         ".", color="orange", markersize=20)

    for node in model.marble_eater_nodes:
        assert torch.numel(node.pos) == 2
        point_coords = node.pos.detach().numpy().reshape((2))
        eaters, = ax.plot(
            point_coords[0], point_coords[1], "x", color="black", markersize=7)

    model_has_marbles = (len(model.marbles) > 0)
    for marble in model.marbles:
        assert torch.numel(marble.pos) == 2
        point_coords = marble.pos.detach().numpy().reshape((2))
        marbles, = ax.plot(
            point_coords[0], point_coords[1], ".", color="lime", markersize=10)

    if model_has_marbles:
        ax.legend([nodes, (nodes, eaters), marbles],
                  ["Nodes", "MarbleEaterNodes", "Marbles"])
    else:
        ax.legend([nodes, (nodes, eaters)], ["Nodes", "MarbleEaterNodes"])

    ax.set_xlabel("$x \\rightarrow$")
    ax.set_ylabel("$y \\rightarrow$")

    if not ax_provided:
        return fig, ax
