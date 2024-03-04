
"""
    applywhere!(ps, model; apply::Function, where::Function)    

Traverse through a `Lux`'s zipped `model` and parameters `ps`, checking
if `where` is satisfied for each layer/chain and if so apply function `apply`.
"""
function applywhere!(ps, model; apply::Function, where::Function)
    if where(model)
        return apply(ps, model)
    end
    if !hasproperty(model, :layers)
        return nothing
    end
    # remove one layer of nesting if needed
    if hasproperty(model.layers, :layers)
        return applywhere!(ps, model.layers; apply, where)
    end
    for (m, p) in zip(model.layers, ps)
        applywhere!(p, m; apply, where)
    end
end