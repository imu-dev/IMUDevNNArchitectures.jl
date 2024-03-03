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
using Flux
using IMUDevNNArchitectures

model = Chain(
    Dense(10, 5),
    PReLU(0.25f0),
    Dense(5, 2),
    softmax)
```
"""
struct PReLU{T}
    α::Vector{T}
end

Flux.@functor PReLU

PReLU(α::Real=0.25f0) = PReLU([α / 1])

prelu(x, α) = max.(0, x) .+ α .* min.(0, x)

(act::PReLU)(x) = prelu(x, act.α)