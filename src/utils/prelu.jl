"""
    PReLU(α::Real=0.25f0)

Parametric Rectified Linear Unit (PReLU). Activation function of the form:

```math
\\begin{cases}
    x & \\text{if } x > 0, \\\\
    αx & \\text{otherwise},
\\end{cases}
```

where `α` is a learnable parameter.

# Example

```julia
using Lux
using IMUDevNNArchitectures

model = Chain(Dense(10 => 5),
              PReLU(0.25f0),
              Dense(5 => 2),
              softmax)
```
"""
struct PReLU{F} <: Lux.AbstractExplicitLayer
    init_α::F
end

function PReLU(α::Real=0.25f0)
    λ() = α / 1
    return PReLU{typeof(λ)}(λ)
end

l = PReLU(0.25f0)

Lux.initialparameters(::AbstractRNG, l::PReLU) = (; α=l.init_α())
Lux.initialstates(::AbstractRNG, l::PReLU) = NamedTuple()

Lux.parameterlength(::PReLU) = 1
Lux.statelength(::PReLU) = 0

function (l::PReLU)(x::AbstractArray, ps, st::NamedTuple)
    y = max.(0, x) .+ ps.α .* min.(0, x)
    return y, st
end