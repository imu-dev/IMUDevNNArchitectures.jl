struct PReLU{T}
    α::Vector{T}
end

Flux.@functor PReLU

PReLU(α::Real=0.25f0) = PReLU([α / 1])

prelu(x, α) = max.(0, x) .+ α .* min.(0, x)

(act::PReLU)(x) = prelu(x, act.α)