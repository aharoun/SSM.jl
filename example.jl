using Revise
using SSM
using Random
using Plots

# Cast  arima model with empty parameters
# arima(p,d,q)
aSim    = arima(2, 1, 1)
# fill the parameters
aSim.ϕ  = [0.5, -0.3];      # AR coeff
aSim.θ  = [0.1];            # MA coeff
aSim.σ2 = [0.1];            # variance of error term
aSim.c  = [0.1];            # constant term
println(aSim)
# or use arima(ϕ, θ, σ², c, d) notation
# aSim = arima(ϕ = [NaN, 0.1], θ = [0.2], σ2 = [0.1], c = [0.2], d = 1)

# simulate arima with sample size 500 with optional random number generator seed = seed
Random.seed!(2017);    # fixing seed for reproducibility
y = simulate(aSim, 500)

# initialize arima model with empty parameters and estimate
# all parameters with NaN will be estimated
a = arima(2, 1, 1)
aEst, res, std = estimate(a, y);


# we can also estimate a subset of parameters
a      = arima(2, 1, 1)
a.ϕ[1] = 0.5;                   # first AR coefficient is fixed at 0.5
println(a)

aEst, res, std = estimate(a, y);   # rest of the parameters will be estimated

# Forecast for the next 10 periods
# yF: point forecast, varF: variance of the forecast
yF, varF = forecast(aEst, y, 10)

plot(1:20, [y[end-9:end];yF])
plot!(10:20, [y[end];yF + sqrt.(varF)])
plot!(10:20, [y[end];yF - sqrt.(varF)])


#= 
We can also write our own model and use the state space framework to simulate data and estimate it.

Let say we want to create a model with parameters `par1` and `par2` each of which can be vectors. 

mutable struct newModel{T} <: AbstractTimeModel
	par1 :: Array{T,1}
	par2 :: Array{T,1}
	... other arguments

	estimableParamField ::  NTuple{2, Symbol} = (:par1, :par2)
end

The last field `estimableParamField` is required to indicate which fields of the model type are estimable. 
We can fix some of the entries of `par1` and `par2` and leave others as NaN. The ones with NaN will be considered as unknown and wil be estimated when the model is passed to `estimate`.

We also need to create a mapping from our model to its state space representation via `StateSpace`.

function StateSpace(a :: newModel)
	...
	... map model `a` parameters to state space matrices
	...

	StateSpace(A, B, C, G, R, H, S, a)
end

  State Space Model representation
  y(t) = A + B×s(t) + u
  s(t) = C + G×s(t-1) + R×ep
  u  ~ N(0,H)
  ep ~ N(0,S)


Now, like arima example, initialize an instance of the model, fix some of the parameters if you like, then pass it to
`estimate` with some data. Hopefully it will estimate.

=#


#end
