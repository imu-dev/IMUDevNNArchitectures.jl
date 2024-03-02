module IMUDevNNArchitectures

using Flux

include(joinpath("utils", "prelu.jl"))
include("resnet_1d.jl")
include("tcn.jl")

export PReLU, prelu

export ResNet1d, FCResNetOutput, FCParams, BasicBlock

export tcn, tcn_block

end
