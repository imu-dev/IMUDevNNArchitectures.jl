# Reference implementations
# https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
# https://github.com/Sachini/ronin/blob/master/source/model_resnet1d.py

module ResNet1d

using Lux
using ..IMUDevNNArchitectures: applywhere!

"""
    inputblock(in_out::Pair{Int,Int})

Input block for ResNet1d. It consists of a `convolutional layer`, a `batch
normalization` layer, and a `max pooling` layer.
"""
function inputblock(in_out::Pair{Int,Int})
    _, out = in_out
    return Chain(Conv((7,), in_out; stride=2, pad=SamePad(), use_bias=false),
                 BatchNorm(out, relu),
                 MaxPool((3,); stride=2, pad=SamePad()))
end

"""
    basicblock(kernel_size::Int, in_out::Pair{Int,Int}; stride::Int)

A `basic block` for ResNet1d. It consists of two `convolutional layers` with
`batch normalization` and a `skip connection`. `ResNet1d` will stack these
`basic blocks` to form `residual groups` (see also [`residualgroup`](@ref)).

# Arguments
- `kernel_size::Int`: The size of the kernel for the convolutional layers.
- `in_out::Pair{Int,Int}`: The input and output channels of the block.
- `stride::Int`: The stride for the first convolutional layer.
"""
function basicblock(kernel_size::Int, in_out::Pair{Int,Int}; stride::Int=1)
    _, out = in_out
    m = Chain(Conv((kernel_size,), in_out; stride, pad=SamePad(), use_bias=false),
              BatchNorm(out, relu),
              Conv((kernel_size,), out => out; pad=SamePad(), use_bias=false),
              BatchNorm(out); name="basicblock_main")
    act = WrappedFunction(Base.Fix1(broadcast, relu))
    if stride > 1
        return Chain(Parallel(+, m, downsampler(in_out; stride)), act)
    end
    return Chain(SkipConnection(m, +), act)
end

"""
    downsampler(in_out::Pair{Int,Int}; stride::Int)

A downsampler block for ResNet1d's `basic block`. It downsamples the input to a
size that makes a `skip connection` possible.

# Arguments
- `in_out::Pair{Int,Int}`: The input and output channels of the ResNet1d's `basic block`.
- `stride::Int`: The stride for downsampling.
"""
function downsampler(in_out::Pair{Int,Int}; stride::Int)
    stride > 1 || throw(ArgumentError("Stride must be greater than 1 when downsampling"))
    _, out = in_out
    return Chain(Conv((1,), in_out; stride, use_bias=false),
                 BatchNorm(out))
end

"""
    zero_init!(model::Chain, parameters)
    
Zero-initialize the last BatchNormalization in each residual branch of the model
`m`. This improves the model by 0.2~0.3% according to
https://arxiv.org/abs/1706.02677.
"""
function zero_init!(model::Chain, parameters)
    return applywhere!(parameters,
                       model;
                       apply=(ps, m) -> ps.layer_4.scale .= 0.0,
                       where=(m) -> hasproperty(m, :name) && m.name == "basicblock_main")
end

"""
    residualgroup(kernel_size::Int, in_out::Pair{Int,Int};
                  group_size::Int, stride::Int=1)

Residual group for ResNet1d. It consists of `basic block`s stacked together.
Only the first `basic block` in the group can have a `stride` greater than 1.

# Arguments
- `kernel_size::Int`: The size of the kernel in each `basic block`.
- `in_out::Pair{Int,Int}`: The input and output channels of the group.
- `group_size::Int`: The number of `basic block`s in the group.
- `stride::Int`: The stride for the first `basic block` in the group.
"""
function residualgroup(kernel_size::Int, in_out::Pair{Int,Int};
                       group_size::Int, stride::Int=1)
    _, out = in_out
    return Chain(basicblock(kernel_size, in_out; stride),
                 [basicblock(kernel_size, out => out) for _ in 2:group_size]...)
end

"""
    in_out_sizes(base_size::Int, schedule::Vector{Int})

Defines the input and output sizes for the `residual groups` in ResNet1d
by scaling the `base_size` by the `schedule`.
"""
function in_out_sizes(base_size::Int, schedule::Vector{Int})
    out_sizes = base_size .* schedule
    in_sizes = vcat([base_size], out_sizes[1:(end - 1)])
    return [in => out for (in, out) in zip(in_sizes, out_sizes)]
end

"""
    incrpow2_schedule(num_groups::Int)

Defines the `schedule` for the sizes of the `residual groups` in ResNet1d as
increasing powers of 2.
"""
function incrpow2_schedule(num_groups::Int)
    num_groups > 0 || throw(ArgumentError("Number of groups must be greater than 0"))
    return 2 .^ (collect(1:num_groups) .- 1)
end

"""
residualblock(base_plane::Int, group_sizes, strides; kernel_size::Int=3)

Defines the `residual block` for ResNet1d. It consists of `residual group`s
stacked together.

# Arguments
- `base_plane::Int`: Hyperparameter based on which the number of channels
  in each `residual group` is going to be computed.
- `group_sizes`: The number of `basic block`s in each `residual group`.
- `strides`: The stride for the first `basic block` in each `residual group`.
- `kernel_size::Int`: The size of the kernel in each `basic block`.
"""
function residualblock(base_plane::Int, group_sizes, strides; kernel_size::Int=3)
    in_outs = in_out_sizes(base_plane, incrpow2_schedule(length(group_sizes)))

    groups = [residualgroup(kernel_size, in_out; group_size, stride)
              for (in_out, group_size, stride) in zip(in_outs, group_sizes, strides)]

    return Chain(groups...)
end

function out_num_channels(residualblock::Chain)
    last_basicblock = residualblock[end - 1]
    last_batchnorm_layer = last_basicblock.layers[end]
    num_channels = last_batchnorm_layer.chs
    return num_channels
end

"""
    transition_layer(in_out::Pair{Int,Int})

Transition layer for ResNet1d that can be stuck between the `residual block`
and an `output layer`. It consists of a `convolutional layer` and a
`batch normalization` layer.
"""
function transition_layer(in_out::Pair{Int,Int})
    _, out = in_out
    return Chain(Conv((1,), in_out; use_bias=false),
                 BatchNorm(out))
end

"""
    output_layer(in_out::Pair{Int,Int};
                 hidden::Int=1024,
                 dropout::Float64=0.5)
            
Output layer for ResNet1d. `hidden` is the size of the fully connected hidden
layers.
"""
function output_layer(in_out::Pair{Int,Int};
                      hidden::Int=1024,
                      dropout::Float64=0.5)
    in, out = in_out
    return Chain(FlattenLayer(),
                 Dense(in, hidden, relu),
                 Dropout(dropout),
                 Dense(hidden, hidden, relu),
                 Dropout(dropout),
                 Dense(hidden, out))
end

"""
    outputblock(in_out::Pair{Tuple{Int,Int},Int};
                hidden::Int=1024,
                dropout::Float64=0.5,
                transition_size::Int=0)

Output block for ResNet1d. It can have a `transition layer` between the
`residual block` and the `output layer`.

# Arguments
- `in_out::Pair{Tuple{Int,Int},Int}`: The input and output sizes of the block.
- `hidden::Int`: The size of the fully connected hidden layers in the `output layer`.
- `dropout::Float64`: The dropout rate for the `output layer`.
- `transition_size::Int`: The size of the `transition layer`.

!!! note
    The `in_out` has a format `(num_channels, time_dim) => output_dim`.
"""
function outputblock(in_out::Pair{Tuple{Int,Int},Int};
                     hidden::Int=1024,
                     dropout::Float64=0.5,
                     transition_size::Int=0)
    in, out = in_out

    if transition_size > 0
        return Chain(transition_layer(in[1] => transition_size),
                     output_layer(transition_size * in[2] => out;
                                  hidden,
                                  dropout))
    end
    return output_layer(prod(in) => out; hidden, dropout)
end

"""
    resnet(in_out::Pair{Tuple{Int,Int},Int}, group_sizes;
           strides=vcat([1], fill(2, length(group_sizes) - 1)),
           base_plane::Int=64, kernel_size::Int=3,
           outputblock_builder=(in_out) -> outputblock(in_out;
                                                       hidden=1024,
                                                       dropout=0.5,
                                                       transition_size=0))

ResNet1d model (see the
[original ResNet publication](https://arxiv.org/abs/1512.03385)). It consists of
an `input block`, a `residual block`, and an `output block`. The
`residual block` is defined by the `group_sizes`, `strides`, `base_plane` and
`kernel_size`. The `output block` is often custom built, but a default one is
provided. `in_out` argument in the `outputblock_builder` is a pair of the form
`(num_channels, time_dim) => output_dim`.

# Arguments
- `in_out::Pair{Tuple{Int,Int},Int}`: The input and output sizes of the model.
- `group_sizes`: The number of `basic block`s in each `residual group`.
- `strides`: The stride for the first `basic block` in each `residual group`.
- `base_plane::Int`: Hyperparameter based on which the number of channels
  in each `residual group` is going to be computed.
- `kernel_size::Int`: The size of the kernel in each `basic block`.
- `outputblock_builder`: A function that builds the `output block` of the model.

!!! note
    The `in_out` has a format `(num_channels, time_dim) => output_dim`.
"""
function resnet(in_out::Pair{Tuple{Int,Int},Int}, group_sizes;
                strides=vcat([1], fill(2, length(group_sizes) - 1)),
                base_plane::Int=64, kernel_size::Int=3,
                outputblock_builder=(in_out) -> outputblock(in_out;
                                                            hidden=1024,
                                                            dropout=0.5,
                                                            transition_size=0))
    in, out = in_out

    b1 = inputblock(in[2] => base_plane)
    b2 = residualblock(base_plane, group_sizes, strides; kernel_size)

    inputblock_strides = [2, 2]
    time_dim = compute_time_dimension(in[1], [inputblock_strides..., strides...])
    num_channels = out_num_channels(b2)
    b3 = outputblock_builder((num_channels, time_dim) => out)

    return Chain(b1, b2, b3)
end

"""
    compute_time_dimension(starting::Int, strides)

Compute the time dimension of the output tensor given the starting time
dimension and the strides of the layers in the model.
"""
function compute_time_dimension(starting::Int, strides)
    return foldl((prev, stride) -> div(prev, stride, RoundUp), strides;
                 init=starting)
end

# Export only the main function
export resnet

end