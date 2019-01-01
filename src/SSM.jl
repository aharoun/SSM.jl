module SSM

using LinearAlgebra
using Distributions
using Optim
using Statistics
using Random

import Base.show


include("types.jl")
include("statespace.jl")
include("arima.jl")


export StateSpace, logLike_Y, arima, simulate, estimate, parseObjFunc  

end # module

