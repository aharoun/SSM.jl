using Revise
using SSM
using BenchmarkTools
using Plots
using CSV


# Construct arima model and simulate
# arima(ϕ, θ, σ², d)
aSim = arima([0.6, -0.3, 0.1], [0.2, -.1], 0.1, 1)

y = simulate(aSim,500)

# initialize arima model with empty parameters and estimate
# arima(p,d,q)
a = arima(3,0,2)
@time aEst, res, std = estimate(a, y);

# we can directly create and estimate a state space model

s = StateSpace(aEst)
s.model = ssmGeneric()
@time sEst, res = estimate(s, y);


