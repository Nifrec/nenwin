Informal logbook of my (Lulof Pirée) thought process while working on the implementation.
This is mostly to preserve my own sanity.

## Particle backprop problem (13-02-2021)
Backpropagating though particles does not always produce gradients in the
parameters of the particles. They just remain `None`. This was supposed to be working and tested, but it appears there is still a findamental bug in the particles.

### Approach:
Writing more testcases for the particles. First for the most simple particles in the class hierarchy, and build that up to the Nodes and Marbles.

Observations:
* The problem is not that they are `nn.Parameter` instances.
* Directly returning and setting a reference to `self.__pos` does also not solve it.

### Update 15-02-2021
I forgot that the `.init_pos` is optimized and not the `.pos`. 
The gradients for the `.init_pos` do seem to work in all cases.
Hence the problem seems to be specific to the `.mass`.
I will add testcases to the `PhysicalParticle`, which is the oldest ancestor with `.mass`.

Further experimentation showed something strange. When I add a parameter `__mass`, 
it will not collect any gradients.
When I add a second variable that is `self.__working_mass = 1 * self._mass`, 
and return this in the getter (and hence also in the computation of the loss),
then `self.__working_mass` will also not collect any gradients, 
*but `self.__mass` will!*
I do not understand this behavior. 
At least it is a workaround,
but I should probably ask this on the PyTorch forum tomorrow.