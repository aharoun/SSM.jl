module SSM

using LinearAlgebra
using Statistics: var, mean
using Calculus
using NamedArrays
using NLopt

import Base.show

include("statespace.jl")
include("arima.jl")


export StateSpace, AbstractTimeModel, ssmGeneric, arima,
arima, simulate, estimate, forecast, aicbic,aic, bic,_estimate,negLogLike! 
end # module

