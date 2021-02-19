Informal logbook of my (Lulof Pirée) thought process 
while working on the implementation.
This is mostly to preserve my own sanity.

## Particle backprop problem (13-02-2021)
Backpropagating though particles does not always produce gradients in the
parameters of the particles. They just remain `None`. 
This was supposed to be working and tested, 
but it appears there is still a findamental bug in the particles.

### Approach:
Writing more testcases for the particles. 
First for the most simple particles in the class hierarchy, 
and build that up to the Nodes and Marbles.

Observations:
* The problem is not that they are `nn.Parameter` instances.
* Directly returning and setting a reference to 
    `self.__pos` does also not solve it.

### Update 15-02-2021
I forgot that the `.init_pos` is optimized and not the `.pos`. 
The gradients for the `.init_pos` do seem to work in all cases.
Hence the problem seems to be specific to the `.mass`.
I will add testcases to the `PhysicalParticle`, 
which is the oldest ancestor with `.mass`.

Further experimentation showed something strange. 
When I add a parameter `__mass`, 
it will not collect any gradients.
When I add a second variable that is `self.__working_mass = 1 * self._mass`, 
and return this in the getter (and hence also in the computation of the loss),
then `self.__working_mass` will also not collect any gradients, 
*but `self.__mass` will!*
I do not understand this behavior. 
At least it is a workaround,
but I should probably ask this on the PyTorch forum tomorrow.

### Update 16-02-2021
I have narrowed the inconsistency down. When having two variables,
one 'loose' variable and another an attribute of a nn.Module, 
and applying copy() and some operations to both, then after backpropagation
only the gradient of the loose variable will be set. 
I asked a question on the PyTorch forum for this.

```python
import torch
import torch.nn as nn
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.__my_parameter = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    @property
    def my_parameter(self) -> torch.Tensor:
        return self.__my_parameter.clone()

v = torch.tensor([1.0], requires_grad=True)
my_module = MyModule()
loss = torch.sum(v.clone() + my_module.my_parameter)
loss.backward()
print(v.grad, my_module.my_parameter.grad)
```
output:
```python
tensor([1.]) None
```

Well, I discovered it works when writing instead:
```python
print(v.grad, my_module._MyModule__my_parameter.grad)
```
This seems as the situation where I started, but it works now.
Rather confusion but 'Eind goed, al goed'?

## Emitter backprop problem (16-02-2021)
**Observation**: using torchviz, it becomes clear that:
* The mass of the output Marbles does not have the supposed gradient tree
* The stored_mass of the Emitter does.
So it seems that it goes wrong at the creation of the new Marble.

The expression
```python
(self.__stored_mass/self.__stored_mass.item())*self.prototype.mass.item()
```
does seem to have the correct gradient dependency graph.

Indeed, further experimentation revealed the location of the bug:

**The copy() method of Marbles does discard the computational graph!**

### Update 17-02-2021

The following should print `True` but it raises an error instead:

```python
some_marble = Marble(ZERO, ZERO, ZERO, 10, None, None, 0, 0, 0, 0)


v = torch.tensor([1.0], requires_grad=True)

loss = torch.sum(v + some_marble.mass)
loss.backward()

another_marble = some_marble.copy()

print(torch.allclose(some_marble.mass.grad, another_marble.mass.grad))
```
The best approach is to add similar scripts as testcases 
to all types of particles 
(except `Particle` which does not have learnable parameters).

`InitialValueParticle.copy()` does pass values containing gradients into
the `__init__()` of the new object. It seems they are discarded during init?
Let's add a testcase for that.

Found where it goed wrong: `torch.nn.Parameter(input)` discards 
`input.grad`, and returns a clone with `None` as grad.

## Update 19-02-2021
A simple fix is the roughly following 
(for copying a vector to a param and keeping grad):
```python
    output = nn.Parameter(input_vect.clone())
    output.grad = input_vect.grad
```
This works and passes all my tests.

The MarbleEmitter is still not working, but I see I made the same mistake
(using `Parameter` without coping the grad manually).

### Wait a moment
I am coping grads during forward propagation. 
But they will only be defined during backpropagation.
That does not sound useful, `grad_fn` needs to be copied instead? 
But that is not possible in PyTorch.

An illustration:
```python
a = torch.tensor([1.0], requires_grad=True)
b = nn.Parameter(a, requires_grad=True)
b.grad = a.grad

loss = torch.tensor([2.0]) * b
b.backward()
print(b.grad, a.grad, sep="\n")
```
output:
```python
tensor([1.])
None
```
So I've basically been cheating my own testcases without being aware of it
(and been using bad testcases in the first place).
How could I be so naiv?

Either way, time to remove the cheats and add better testcases, 
and start over again...