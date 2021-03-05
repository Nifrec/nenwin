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

Plot of gravity functions such as
Newton's gravity formula, simplified to 1/r, and a similar-shaped
non-polynomial function that has a finite limit at r -> 0.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
from nenwin.attraction_functions.attraction_functions import NewtonianGravity
from nenwin.attraction_functions.attraction_functions import Gratan

def clipped_gravity(x: np.ndarray):
    x = x.copy()
    x[abs(x) > 3] = 0
    x[x != 0] = abs(1/x[x != 0]**2)
    return x


X = np.concatenate([np.arange(-100, 0, 1),[-0.9, 0.9], np.arange(1, 100, 1)])
X = np.concatenate([np.arange(-10, 0, 0.1), np.arange(0.1, 10.1, 0.1)])
# plt.plot(X, 0.2*abs(1/X**2), label="Absolute value of gravity")
plt.plot(X, 0.2*clipped_gravity(X), label="clipped_gravity")
# plt.plot(X, 1 - abs(np.tanh(X)), label="$1 - |\\tanh(x)|$")
plt.legend()
plt.xlabel("Distance")
plt.ylabel("Attraction strength")
plt.show()



