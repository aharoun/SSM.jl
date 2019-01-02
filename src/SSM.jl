module SSM

using LinearAlgebra
using Distributions
using Optim
using Statistics
using Random
using Calculus

import Base.show


include("types.jl")
include("statespace.jl")
include("arima.jl")


export StateSpace, ssmGeneric, nLogLike, arima, simulate, estimate, ssmNegLogLike!

end # module

