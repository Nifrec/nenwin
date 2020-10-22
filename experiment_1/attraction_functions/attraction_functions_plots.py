"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
October 2020

Plot of Newton's gravity formula, simplified to 1/r, and a similar-shaped
non-polynomial function that has a finite limit at r -> 0.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np

X = np.concatenate([np.arange(-100, 0, 1),[-0.9, 0.9], np.arange(1, 100, 1)])
X = np.concatenate([np.arange(-10, 0, 0.1), np.arange(0.1, 10.1, 0.1)])
plt.plot(X, 0.2*abs(1/X), label="Absolute value of gravity")
plt.plot(X, 1 - abs(np.tanh(X)), label="$1 - |\\tanh(x)|$")
plt.legend()
plt.xlabel("Distance")
plt.ylabel("Attraction strength")
plt.show()



