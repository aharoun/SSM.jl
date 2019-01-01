using Revise
using SSM
using BenchmarkTools
using Plots
using CSV


# Construct arima model and simulate
# arima(ϕ, θ, σ², d)
a = arima([0.5, -0.3, 0.1], [0.2, -0.3], 0.1, 0)

y = simulate(a,500)

# initialize arima model with empty parameters and estimate
a = arima(3,0,2)
@time aEst, res = estimate(a, y);

# we can directly create and estimate a state space model

s = StateSpace(a)
s.model = ssmGeneric()
@time sEst, res = estimate(s, y);


