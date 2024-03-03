using Documenter
using IMUDevNNArchitectures

makedocs(; sitename="IMUDevNNArchitectures",
         format=Documenter.HTML(),
         modules=[IMUDevNNArchitectures],
         checkdocs=:exports,
         pages=["Home" => "index.md",
                "Manual" => ["Foreword" => joinpath("pages", "general_note.md"),
                             "ResNet" => joinpath("pages", "resnet.md"),
                             "TCN" => joinpath("pages", "tcn.md"),
                             "utilities" => joinpath("pages", "utils.md")]])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
deploydocs(; repo="github.com/imu-dev/IMUDevNNArchitectures.jl.git")
