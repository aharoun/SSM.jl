using Revise
using SSM
using BenchmarkTools
using Plots
using CSV



a = arima(3,0,2)
a.ϕ[1:3] = [.5,-.3, .1]
a.θ[1:2] = [.2 -0.3]
a.σ2 = .1

y = simulate(a,500)

@time res, s = estimate(arima(3,0,2), y);


