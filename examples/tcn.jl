using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUDevNNArchitectures", "examples"))
using Lux
using IMUDevNNArchitectures
using Random

rng = Random.default_rng()
Random.seed!(rng, 0)

model = Chain(tcn(6 => 36;
                  channels=[32, 64, 128, 256, 72],
                  kernel_size=3,
                  dropout=0.2),
              Conv((1,), 36 => 2),
              Dropout(0.2))

ps, st = Lux.setup(rng, model)

numtimepoints = 200
numfeatures = 6
batchsize = 32
input = rand(Float32, numtimepoints, numfeatures, batchsize)

# pass input through the model to see if layer sizes go well together
model(input, ps, st)