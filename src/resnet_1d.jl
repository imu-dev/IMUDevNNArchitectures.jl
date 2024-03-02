# Reference implementations
# https://github.com/hsd1503/ResNet1d/blob/master/ResNet1d.py
# and
# https://github.com/Sachini/ronin/blob/master/source/model_ResNet1d.py

struct BasicBlock
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
    downsampler::Union{Chain,Nothing}
end

Flux.@functor BasicBlock

function BasicBlock(kernel_size::Int, in_out::Pair{Int,Int};
                    stride::Int=1, pad=kernel_size ÷ 2)
    in, out = in_out
    # assertion coming out of construction of ResNet1d
    @assert (in == out && stride == 1) || (in != out && stride != 1)

    has_downsampling = stride != 1
    downsampler = if has_downsampling
        Chain(Conv((1,), in_out; stride, bias=false),
              BatchNorm(out))
    else
        nothing
    end
    return BasicBlock(Conv((kernel_size,), in_out; stride, pad=(pad,), bias=false),
                      BatchNorm(out, relu),
                      Conv((kernel_size,), out => out; pad=(pad,), bias=false),
                      BatchNorm(out),
                      downsampler)
end

function (m::BasicBlock)(x)
    y = x |> m.conv1 |> m.bn1 |> m.conv2 |> m.bn2
    if isnothing(m.downsampler)
        return relu.(x + y)
    else
        return relu.(m.downsampler(x) + y)
    end
end

struct FCResNetOutput
    transition::Union{Nothing,Chain}
    output_layer::Chain
end

Flux.@functor FCResNetOutput

@kwdef struct FCParams
    fc_dim::Int = 1024
    dropout::Float64 = 0.5
    trans_planes::Int = 0
end

function FCResNetOutput(in_size::Tuple{Int,Int}, out::Int, θ::FCParams=FCParams())
    has_transition_layer = θ.trans_planes > 0
    transition, fc_in = if has_transition_layer
        Chain(Conv((1,), in_size[1] => θ.trans_planes; bias=false),
              BatchNorm(θ.trans_planes)), θ.trans_planes * in_size[2]
    else
        nothing, prod(in_size)
    end
    output_layer = Chain(Dense(fc_in, θ.fc_dim, relu),
                         Dropout(θ.dropout),
                         Dense(θ.fc_dim, θ.fc_dim, relu),
                         Dropout(θ.dropout),
                         Dense(θ.fc_dim, out))
    return FCResNetOutput(transition, output_layer)
end

function (m::FCResNetOutput)(x)
    if !isnothing(m.transition)
        x = m.transition(x)
    end
    return x |> Flux.flatten |> m.output_layer
end

function aux_dim(in::Int, strides)
    return foldl((prev, stride) -> div(prev, stride, RoundUp), strides; init=in)
end

@kwdef struct ResNet1d
    input_block::Chain
    residual_groups::Chain
    output_block::FCResNetOutput
end

Flux.@functor ResNet1d

(m::ResNet1d)(x) = x |> m.input_block |> m.residual_groups |> m.output_block

function input_block(::Type{ResNet1d}, in_out::Pair{Int,Int})
    _, out = in_out
    return Chain(Conv((7,), in_out; stride=2, pad=(3,), bias=false),
                 BatchNorm(out, relu),
                 MaxPool((3,); stride=2, pad=(1,)))
end

function residual_group(kernel_size::Int, in_out::Pair{Int,Int};
                        group_size::Int, stride::Int=1)
    _, out = in_out
    return Chain(BasicBlock(kernel_size, in_out; stride),
                 [BasicBlock(kernel_size, out => out) for _ in 2:group_size]...)
end

function residual_groups(::Type{ResNet1d},
                         base_plane::Int,
                         group_sizes;
                         kernel_size::Int=3)
    out_planes = base_plane .* (2 .^ (eachindex(group_sizes) .- 1))
    in_planes = vcat([base_plane], out_planes[1:(end - 1)])
    strides = out_planes .÷ in_planes

    groups = [residual_group(kernel_size, in => out; group_size, stride)
              for (in, out, group_size, stride) in
                  zip(in_planes, out_planes, group_sizes, strides)]

    return Chain(groups...), out_planes[end], strides
end

function ResNet1d(in_size_out::Pair{Tuple{Int,Int},Int}, residual_group_sizes;
                  base_plane::Int=64, zero_init_residual::Bool=false,
                  residual_kernel_size::Int=3,
                  fc_params::FCParams=FCParams())
    in_size, out = in_size_out
    b1 = input_block(ResNet1d, in_size[2] => base_plane)
    b2, last_plane_size, strides = residual_groups(ResNet1d,
                                                   base_plane,
                                                   residual_group_sizes;
                                                   kernel_size=residual_kernel_size)
    input_block_strides = [2, 2]
    d = aux_dim(in_size[1], [input_block_strides..., strides...])
    b3 = FCResNetOutput((last_plane_size, d), out, fc_params)

    out = ResNet1d(; input_block=b1,
                   residual_groups=b2,
                   output_block=b3)
    init!(out, zero_init_residual)
    return out
end

function init!(m::ResNet1d, zero_init_residual::Bool)
    for mod in Flux.modules(m)
        if isa(mod, Conv)
            mod.weight .= Flux.kaiming_normal(size(mod.weight)...)
        elseif isa(mod, BatchNorm)
            mod.γ .= 1.0
            mod.β .= 0.0
        elseif isa(mod, Dense)
            mod.weight .= randn(Float32, size(mod.weight)) .* 0.01f0
            mod.bias .= 0.0
        end
    end

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual
        for mod in Flux.modules(m)
            if isa(mod, BasicBlock)
                mod.bn2.γ .= 0.0
            end
        end
    end
    return nothing
end