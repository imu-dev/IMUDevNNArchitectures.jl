# Residual Network (ResNet 1d)

We implement 1d [Residual Networks](https://arxiv.org/abs/1512.03385).

!!! note
    ResNet is an example of a `sequence-to-point`, `non-recurrent` architecture. See [foreword](#) for more details.

```@docs
resnet
```

```@docs
IMUDevNNArchitectures.ResNet1d.outputblock
```

```@docs
IMUDevNNArchitectures.ResNet1d.transition_layer
IMUDevNNArchitectures.ResNet1d.output_layer
```

```@docs
IMUDevNNArchitectures.ResNet1d.init!
IMUDevNNArchitectures.ResNet1d.zero_init!
```


```@docs
IMUDevNNArchitectures.ResNet1d.inputblock
IMUDevNNArchitectures.ResNet1d.residualblock
```

```@docs
IMUDevNNArchitectures.ResNet1d.residualgroup
IMUDevNNArchitectures.ResNet1d.basicblock
```