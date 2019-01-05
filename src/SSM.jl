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
nLogLike, arima, simulate, estimate, forecast

end # module

