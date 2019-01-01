module StateSpaceModeling

using LinearAlgebra
using SparseArrays
using Distributions
using Optim
using Statistics
using Random



include("functions.jl")


export StateSpace, logLike_Y, arima, simulate, estimate 

end # module
