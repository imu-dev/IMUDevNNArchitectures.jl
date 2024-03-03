using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUDevNNArchitectures", "examples"))
using Flux
using IMUDevNNArchitectures

model = Chain(tcn(6 => 36;
                  channels=[32, 64, 128, 256, 72],
                  kernel_size=3,
                  dropout=0.2),
              Conv((1,), 36 => 2),
              Dropout(0.2))

numtimepoints = 200
numfeatures = 6
batchsize = 32
input = rand(Float32, numtimepoints, numfeatures, batchsize)

# pass input through the model to see if layer sizes go well together
model(input)