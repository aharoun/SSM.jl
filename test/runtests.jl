using SSM, Random, Test

# Cast  arima model with empty parameters
aSim    = arima(2, 1, 1)
# fill the parameters
aSim.ϕ  = [0.5, -0.3];      # AR coeff
aSim.θ  = [0.1];            # MA coeff
aSim.σ² = [0.1];            # variance of error term
aSim.c  = [0.1];            # constant term

# simulate arima with sample size 500
# here we can also pass a StateSpace object directly
Random.seed!(2);    # fixing seed for reproducibility
y = simulate(aSim,500)

# all parameters with NaN will be estimated
aEst, estParams, res1 = estimate(arima(2, 1, 1), y);
@test res1.ret==:FTOL_REACHED

# choose lag lenght based on aic or bic (choose the minimum)
aicTable, bicTable = aicbic(arima(2,1,2), y); # this will calculate aic and bic for all models upto arima(pMax,1,qMax)
@test aicTable[1,2] ≈ 277.70904020807035
@test bicTable[2,2] ≈ 286.2058900239874

# Use arima(ϕ, θ, σ², c, d) notation
a = arima(ϕ = [NaN, 0.1], θ = [0.2], σ² = [0.1], c = [0.2], d = 1)

aEst, estParams, res2 = estimate(a, y);   # rest of the parameters will be estimated

@test res2.ret==:FTOL_REACHED
# Forecast 
yF, varF = forecast(aEst, y, 10)


