# Temporal Convolutional Network (TCN)
We implement a [Temporal Convolutional Network](https://arxiv.org/abs/1608.08242)

!!! note
    TCN is an example of a `sequence-to-sequence`, `non-recurrent` architecture. See [classification page](@ref "Classification of NN architectures for temporal data") for more details.

The main constructor is given by:

```@docs
tcn
```

Note that this will define the bulk of your network, but you should still define the output layers. For instance:

## Example

```julia
using Lux
using IMUDevNNArchitectures
using Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(tcn(6 => 36;
                  channels=[32, 64, 128, 256, 72],
                  kernel_size=3,
                  dropout=0.2),
              Conv((1,), 36 => 2),
              Dropout(0.2))

ps, st = Lux.setup(rng, model)

numtimepoints = 200
numfeatures = 6
batchsize = 32
input = rand(Float32, numtimepoints, numfeatures, batchsize)

# pass input through the model to see if layer sizes go well together
model(input, ps, st)
```

