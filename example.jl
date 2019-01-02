using Revise
using SSM

# Cast parametrized arima model
# arima(p,d,q)

aSim    = arima(2, 1, 1)
aSim.ϕ  = [0.5, -0.3];      # AR coeff
aSim.θ  = [0.2];            # MA coeff
aSim.σ2 = [0.1];              # variance of error term

println(aSim)
# or use arima(ϕ, θ, σ², d) notation
# aSim = arima([NaN, 0.1], [0.2], 0.1, 0)

# simulate arima with sample size 500
y = simulate(aSim, 500)

# initialize arima model with empty parameters and estimate
# all parameters with NaN will be estimated
a = arima(2, 1, 1)
@time aEst, res, std = estimate(a, y);

# we can also estimate a subset of parameters
a      = arima(2, 0, 1)
a.ϕ[1] = 0.5 ;           # first AR coefficient is fixed at 0.5
println(a)

aEst, res, std = estimate(a, y)   # rest of the parameters will be estimated
