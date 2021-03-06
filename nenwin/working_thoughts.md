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

*Actually the testcases do not pass, I am confused now. Did I run the wrong ones?*

**At least the way to proceed is clear**: the only method of preserving
grads is to use `clone()`, initializing a new `Parameter` deleted the grads.
Hence only the initial values of `pos`, `vel`, `acc`
 and `mass` should be `Parameter`s,
the working versions just be normal Tensors.
This is needed because `Parameter` attributes 
can only be assigned new `Parameter` instances, which I cannot when I change
the position and the mass etc.

Well, for the mass it is not needed. The specification of Nenwin
does not require the mass to be changeable at runtime,
so I could simply remove the setter altogether.

Plan:
* Change `pos`, `vel`, `acc` setters to not return a `Parameter`
* Remove `mass` getter.

I decided the following:
* I added a method `adopt_parameters()` to adopt parameters 
    from another particle into a `InitialValueParticle`.
    This copies references to the initial values,
    hence gradients will flow back to the original.
* I dropped the requirement for coping grads and dependency graphs during
    init. One can use the new method `adopt_parameters()` to get the
    same result if required.
The disadvantage is that all `Particle` subclasses require modification and
additional testcases.

## Update 22-02-2021
I dropped the constraint (and adapted implementation) that 
`MarbleEaterNode.radius` must be a learnable parameter.
There currently was no way for it to collect gradients.
This is probably something to add later though!

## Update 27-02-2021
One of the difficulties with the `Emitter` is that the mass of the
emitted Marble should propagate back to the `stored_mass` of the `Emitter`.
However, the mass of a Marble is a `torch.nn.Parameter()`, and creating
a new parameter for a tensor resets the computational graph.
Hence it is not possible to both
1. Keep the mass of the Marble trainable (i.e. keep it a Parameter).
2. Propagate back to the `stored_mass` and the `prototype`'s mass.
At first it seemed that the only solution is to give a Marble an `init_mass`,
which is a Parameter, and a 'working' mass, and only set the working mass
when emitting a new Marble. However, 
**The emitted Marble does not need to be trainable!**. 
It does not exist at the start of the algorithm, and when optimizing the other
particles it may even never come into existance. So I am confusing goal
of different parts of the software with each other.

To remove `Parameter`s from a module I needed a newer PyTorch version.
So waiting a lot of time before things to install -- \*sigh\*.

Apparently that module does something different. All the updating effort for nothing... Ah well it needed updating anyway, but preferable while I was working on something else...

The good news is that registered Parameters
can be removed from a module simply with `del`.

## Update 05-03-2021
The mass of a Marble created by an emitter **WILL NEVER COLLECT GRADS**.
Non-leaf tensors do not store grad, and it is not a leaf because this
mass is derived from the `stored_mass` of the emitter 
and the mass of the prototype.

I also noticed that I am making quite a mess by making a distinction between
the `mass` attribute of emitted marbles and 'normal' marbles.
It is clear I need to make the difference more explicit. 
Introducing a new class: `EmittedMarble`!
Hopefully this will finally solve things...

## Update 06-03-2021

To train a NenwinModel, we need to compute a loss function based on the
output of the network (the amount of Marbles eaten by the MarbleEaterNodes), 
and use the derivative of this loss w.r.t. the 
particle's parameters for the optimization algorithm.
Hence the output must be differentiable w.r.t. the Marbles eaten.
So far I forgot to implement this. 

### Idea 1:
Set `num_marbles_eaten <- m.pos / m.pos.item()`.
The the gradient of `num_marbles_eaten` w.r.t. `m.pos`is 1.
This only works if the Marble has actually been eaten. 
Hence it can be used to punish Marbles eaten by the wrong MarbleEaterNode,
but not to make Marbles turn to the location where they should end up.

## Idea 2:
Blame the Marble who is not at the right position. 
If we demand a Marble to be at `x_1`, and it is at `x_2`, then
we can set `|x_1 - x_2|²` as the error (as the loss).

This becomes more difficult when there are multiple Marbles present.
Do we punish all of them? But we may need only one at `x_1`, so
it does not make sense to optimize the architecture to get *all*
Marbles there; this may even be counterproductive to later desired outputs
on the other side of the network.

Only the clostest Marble? This seems to be a better option, 
but not in all cases. For example, a Marble `m_1` may be moving with a high
speed towards `x_1`, but still be a little further away than some Marble
`m_2` that is moving *away* from `x_1`. Blaming `m_1` would seem more
fitting than blaming `m_2`.

But we can take the velocity into account.
We can blame the Marble with the least value of

`x_1 - m.pos + m.vel`

By why weight them evenly?

`x_1 - α*m.pos + β*m.vel`

Here, α and β are not trainable, as any loss function can easily get unbounded scores by increasing/decreasing them to limits. 

We can state that the loss is the minimum

`|x_1 - (m.pos + α*m.vel)|²` over all Marbles `m`. But let's just begin with `α = 0`, i.e. just blaming the closest Marble.