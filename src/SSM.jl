module SSM

using LinearAlgebra
using Statistics: var, mean
using Calculus
using NamedArrays
using NLopt
using ForwardDiff
import Base.show

include("statespace.jl")
include("arima.jl")


export StateSpace, AbstractTimeModel, ssmGeneric, 
arima, simulate, estimate, forecast, aicbic,aic, bic,_estimate,negLogLike!,nLogLike 
end # module

