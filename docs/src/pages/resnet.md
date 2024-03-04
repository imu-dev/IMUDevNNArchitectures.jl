# Residual Network (ResNet 1d)

We implement 1d [Residual Networks](https://arxiv.org/abs/1512.03385).

!!! note
    ResNet is an example of a `sequence-to-point`, `non-recurrent` architecture. See [classification page](@ref "Classification of NN architectures for temporal data") for more details.

## Main functions

The main constructor is given by:

```@docs
resnet
```

The outputblock is often custom built, but we implement a good candidate:

```@docs
IMUDevNNArchitectures.ResNet1d.outputblock
```

Under the hood it comprises of two layers:
- an optional `transition_layer`, and
- an `output_layer`:

```@docs
IMUDevNNArchitectures.ResNet1d.transition_layer
IMUDevNNArchitectures.ResNet1d.output_layer
```

Finally, as a miscellaneous step we might consider applying a special initialization to some `BatchNorm` layers:

```@docs
IMUDevNNArchitectures.ResNet1d.zero_init!
```

### Example
An example implementation could look as follows:

```julia
using Lux
using IMUDevNNArchitectures
using IMUDevNNArchitectures.ResNet1d
using Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = resnet((200, 6) => 2, [2, 2, 2, 2];
               strides=[1, 2, 2, 2],
               base_plane=64, kernel_size=3,
               outputblock_builder=(in_out) -> ResNet1d.outputblock(in_out;
                                                                    hidden=512,
                                                                    dropout=0.5,
                                                                    transition_size=128))

# Define some random input
numtimepoints = 200
numfeatures = 6
batchsize = 32
input = rand(Float32, numtimepoints, numfeatures, batchsize)


ps, st = Lux.setup(rng, model)
# miscellaneous initialization if needed
ResNet1d.zero_init!(model, ps)
# pass through a model
model(input, ps, st)
```


## Details

`ResNet` comprises of the following two main blocks:


```@docs
IMUDevNNArchitectures.ResNet1d.inputblock
IMUDevNNArchitectures.ResNet1d.residualblock
```

where the latter one is a bunch of stacked `residualgroup`s, each comprising of a couple `basicblocks`:

```@docs
IMUDevNNArchitectures.ResNet1d.residualgroup
IMUDevNNArchitectures.ResNet1d.basicblock
```