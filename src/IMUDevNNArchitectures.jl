module IMUDevNNArchitectures

using Flux

include(joinpath("utils", "prelu.jl"))
include("resnet_1d.jl")
include("tcn.jl")

using .ResNet1d: resnet
using .TCN: tcn

export PReLU, prelu
export resnet
export tcn

end
