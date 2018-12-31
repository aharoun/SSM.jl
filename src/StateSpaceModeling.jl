module StateSpaceModeling

using LinearAlgebra
using SparseArrays
using Distributions
using Optim

include("functions.jl")


export StateSpaceModel, logLike_Y, arma, simulate, estimate, initialize, modelTechAdpotion

end # module
