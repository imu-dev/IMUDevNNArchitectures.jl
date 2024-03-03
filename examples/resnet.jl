using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUDevNNArchitectures", "examples"))
using IMUDevNNArchitectures
using IMUDevNNArchitectures.ResNet1d

models = (a=resnet((200, 6) => 2, [2, 2, 2, 2];
                   strides=[1, 2, 2, 2],
                   base_plane=64, kernel_size=3,
                   outputblock_builder=(in_out) -> ResNet1d.outputblock(in_out;
                                                                        hidden=512,
                                                                        dropout=0.5,
                                                                        transition_size=128)),
          b=resnet((200, 6) => 2, [3, 4, 6, 3];
                   strides=[1, 2, 2, 2],
                   base_plane=64, kernel_size=3,
                   outputblock_builder=(in_out) -> ResNet1d.outputblock(in_out;
                                                                        hidden=1024,
                                                                        dropout=0.5,
                                                                        transition_size=128)),
          c=resnet((200, 6) => 2, [3, 4, 23, 3];
                   strides=[1, 2, 2, 2],
                   base_plane=64, kernel_size=3,
                   outputblock_builder=(in_out) -> ResNet1d.outputblock(in_out;
                                                                        hidden=1024,
                                                                        dropout=0.5,
                                                                        transition_size=128)))

numtimepoints = 200
numfeatures = 6
batchsize = 32
input = rand(Float32, numtimepoints, numfeatures, batchsize)

# pass input through a model to see if layer sizes go well together
models.a(input)
models.b(input)
models.c(input)