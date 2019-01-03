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


export StateSpace, AbstractTimeModel, ssmGeneric, arima,
       nLogLike, arima, simulate, estimate, forecast

end # module

