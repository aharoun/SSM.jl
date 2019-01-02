module SSM

using LinearAlgebra
using Distributions: MvNormal
using Optim
using Statistics
using Random
using Calculus

import Base.show

include("statespace.jl")
include("arima.jl")


export StateSpace, ssmGeneric, nLogLike, arima, simulate, estimate, AbstractTimeModel 

end # module

