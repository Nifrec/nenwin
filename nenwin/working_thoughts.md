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

But we can take the velocity into account, and make a *very coarse* estimate of the position in the near future.
We can blame the Marble with the least value of

`x_1 - m.pos + α*m.vel`

Here, α is not trainable, as any loss function can easily get unbounded scores by increasing/decreasing them to limits. 

We can state that the loss is the minimum

`|x_1 - (m.pos + α*m.vel)|²` over all Marbles `m`. 

But let's just begin with `α = 0`, i.e. just blaming the closest Marble.

## Update 13-03-2021
PyTorch is reporting errors in the backpropagation experiment.
Currently it does execute one epoch, including backprop and optimization,
successfully. But it crashed during backprop of the second epoch:
it reports some variable has been modified in-place.
It is difficult to imagine what can be different between the first and
the second epoch.

When not optimizing, so when only computing the gradient,
and keeping `retain_graph = True`, it does not crash.

I found something related on the [PyTorch forum](https://discuss.pytorch.org/t/in-place-operation-error-pytorch1-6-0/97032/7?u=nifrec)

The `_version` attribute of Tensors (the pos, vel, acc, mass and node_stiffness
of the Marble) does turn to 1 in the second epoch:
```python
print(marble.init_pos._version)
print(marble.init_vel._version)
print(marble.init_acc._version)
print(marble.mass._version)
print(marble.node_stiffness._version)
```
gives:
```python
being epoch 0
0
0
0
0
0
end epoch 0
0
0
0
0
0
being epoch 1
1
1
1
1
1
end epoch 1
1
1
1
1
1
```

I tried the same experiment, but now only with a single `Marble`that
has not been encapsulated in a `NenwinModel`:
```python
marble = Marble(marble_pos, zero, zero, mass, NewtonianGravity(), None)
optimizer = torch.optim.Adam(marble.parameters())
step_time=0.5


for epoch in range(25):
    optimizer.zero_grad(set_to_none=True)

    marble.zero_grad(set_to_none=True)
    marble.reset()
    
    t = 0
    while t <= 5:
        marble.update_movement(time_passed=step_time)
        t += step_size

    loss = torch.mean(torch.pow(marble.pos - torch.tensor([0, 0], dtype=torch.float), 2))

    loss.backward()
    del loss
    optimizer.step()
```
Now we get a different error:
```python
RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed. Specify retain_graph=True when calling backward the first time.
```
Also, the second epoch, the init_pos is at `_version` 1, while the mass
stays at version 0. Of course the mass has not been used as there were no
forces applied to the Marble.

### Is it the `_prev_acc` and `_prev_prev_acc`?
They are not supposed to do gradient tracking 
(not sure why I decided that, I now think they should),
but they do: and they may depend on values of `init_acc` from *before*
the last optimizer step!

The following indeed fails:

```python
def test_reset_prev_acc(self):
        """
        When calling reset(), the prev_acc should not depend
        on any value except init_acc!
        """

        particle = setup_simple_particle()
        optim = torch.optim.Adam(particle.parameters())
        for x in range(2):
            particle.update_movement(5)
            particle.pos.backward()
            optim.step()


            particle.zero_grad()
            particle.reset()

        self.assertEqual(particle.init_acc._version,
                         particle._prev_acc._version)
```

Although, PyTorch *wants* the things in the computational graph to have
`_version` 0, which in this case only the `_prev_acc` and `_prev_prev_acc`
have. Now `.clone()` returns a Tensor with `_version` 0, and
both `_prev_acc` and `_prev_prev_acc` are created through a clone.
So it makes sense these values are ok.

Now here some code that works:

```python
mm = MyModule()
optim = torch.optim.Adam(mm.parameters())

for epoch in range(2):
    optim.zero_grad()
    loss = mm(torch.tensor([2.0]))
    loss.backward()
    optim.step()
print(mm.my_param._version)
print(mm(torch.tensor([2.0]))._version)
print(loss._version)
```
with output:
```python
2
0
0
```

Clearly, `my_param` is part of the computational graph.
Yet it has version 2. 
Why does this not raise an error, but does it raise errors for Marbles?

# Running MNIST training
## 17-04-2021
Okay we *finally* got to the point where we can run the training algorithm, 
with backprob, on MNIST. 
As you would expect, it does not work. 
In particular, my machine runs out of memory 
**before completing the very first sample of the very first epoch**. 
Even with only 8 non-output Nodes, 
10 MarbleEaterNodes and *only 10 timesteps per sample.*

One possible explanation is that tracking gradients for 28x28 Marbles 
takes up too much memory. 
This can easily be tested by disabling gradient checking for Marbles, 
which should be possible I suppose. 
In case it is not, gradient checking can be disabled for the whole training. 
Clearly this does not train anything, 
but it can still be used to confirm my hypothesis.

### Disable input Marble gradient tracking
```python
m.requires_grad(False)
```
can simply be used to disable gradient tracking for a single marble `m`.

Disable gradient tracking for the Marbles is not enough
(I still run out of memory before finishing the first sample).
They interact with the Nodes, so gradients for the variables
of the Marbles will still be computed for the parameters of the Nodes.

### Disable all gradients during training
Wrapping the whole training in
```python
with torch.no_grad():
    ...
```
does keep memory usage almost stable. 
It does not increase much after loading the dataset.
However, **the training remains extremely slow**.
It takes over 20 minutes for a single sample 
(after which I terminated the process).
It appears that the O(n²) runtime complexity, 
(where n is the number of particles),
is still far too large to be feasible on MNIST.

### Possible actions:
1. Use a dataset with a much smaller amount of input Marbles.
    + Quick confirmation whether it works.
    - No long-term solution.
1. Evaluate runtime/memory complexity of backprop.
    + Adds scientific value.
    + May provide insight into possible solution.
    - Does not solve anything by itself.
1. Use multiprocessing. 
    Computing the next state for each particle can be done in parallel. Need to re-synchronize after each timestep, but that is O(n). Would have been much easier with multithreading.
    - Some optimizations need to be added at some point.
    - Probably an awful lot of implementation fuss.
    - Quite some overhead. 
    * According to the [PyTorch docs](https://pytorch.org/docs/stable/notes/multiprocessing.html#hogwild),
    `model.share_memory()` can simply be called to share memory. If each sub-process has a set of Marbles of which it knows it should update those, then this may actually work.
    
