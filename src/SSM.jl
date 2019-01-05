module SSM

using LinearAlgebra
using Optim
using Statistics: var
using Calculus
using NamedArrays

import Base.show

include("statespace.jl")
include("arima.jl")


export StateSpace, AbstractTimeModel, ssmGeneric, arima,
nLogLike2, arima, simulate, estimate, forecast, _estimate
end # module

