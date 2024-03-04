module IMUDevNNArchitectures

using Lux
using Random

include(joinpath("utils", "applywhere.jl"))
include(joinpath("utils", "prelu.jl"))
include("resnet_1d.jl")
include("tcn.jl")

using .ResNet1d: resnet
using .TCN: tcn

export applywhere!
export PReLU, prelu
export resnet
export tcn

end
