## Particle backprop problem (13-02-2021)
Backpropagating though particles does not always produce gradients in the
parameters of the particles. They just remain `None`. This was supposed to be working and tested, but it appears there is still a findamental bug in the particles.

### Approach:
Writing more testcases for the particles. First for the most simple particles in the class hierarchy, and build that up to the Nodes and Marbles.

Observations:
* The problem is not that they are `nn.Parameter` instances.
* Directly returning and setting a reference to `self.__pos` does also not solve it.