
# Univariate ARIMA(p,d,q) without any constant for now
mutable struct arima{T<:Real} <: AbstractTimeModel
    p :: Int64		# AR length
    d :: Int64		# Integration order
    q :: Int64		# MA length
    ϕ :: Array{T,1}	# AR coefficients
    θ :: Array{T,1}	# MA coefficients
    σ2:: Array{T,1}	# variance of error term

   estPar :: NTuple{3, Symbol}  # holds the fields of a model type that can be estimated

end

# Initialize arima object with empty parameters
function arima(p::Int64,d::Int64,q::Int64)
    ϕ  = fill(NaN,p)
    θ  = fill(NaN,q)
    σ2 = fill(NaN,1)
    estPar = (:ϕ, :θ, :σ2)
    arima(p, d, q, ϕ, θ, σ2, estPar)
end

# Initialize with parameter vectors, some parameters can be set as NaN. Those can be estimated.
function arima(ϕ::Array{T,1}, θ::Array{T,1}, σ2::Array{T,1}, d::Int64) where T
    p = length(ϕ)
    q = length(θ)

    estPar = (:ϕ, :θ, :σ2)
    arima(p, d, q, ϕ, θ, σ2, estPar)
end

function Base.show(io::IO, a::arima)
    println(io,"ARIMA(",a.p,",",a.d,",",a.q,") Model")
    println(io,"-------------------")
    println(io," AR: ",a.ϕ)
    println(io," MA: ",a.θ)
    println(io," σ2: ",a.σ2)
end


# Cast arima model into state space
# This is for Δᵈy, so y will be differenced d times before estimation
function StateSpace(a::arima)
    m = max(a.p,a.q + 1)

    A = zeros(1)
    B = zeros(1,m)
    B[1] = 1.0

    G = zeros(m,m)
    G[1:a.p,1]   = a.ϕ
    G[1:m-1,2:end] = diagm(0 => ones(m-1))
 
    R = zeros(m,1)
    R[1:1+a.q] = [1.0;a.θ]

    H = zeros(1,1)
    S = fill(a.σ2...,1,1)

    StateSpace(A, B, G, R, H, S, a)

end
# simulate arima model. This can be done generic but MvNormal does not like zero variance
function simulate(a::arima,T::Int64)
    Random.seed!(20)
    ssm = StateSpace(a)
    if !all(isempty.(findEstParamIndex(a)))
      throw("Some parameters are not defined!")
    end

    TT = Int64(round(T*1.5))
    y = zeros(TT,length(ssm.A))

    s  = zeros(size(ssm.G,1))

    @inbounds for t in 1:TT
	s      .= ssm.G*s + ssm.R*rand(MvNormal(ssm.S))
	y[t,:]  = ssm.A + ssm.B*s 
    end

    # above was for Δᵈy, get y now
    for i in 1:a.d
       y = cumsum(y, dims=1)
    end

    # Discard initial draws
    y = y[end-T+1:end,:]

end

# Initialize arima coefficients for estimation
function initializeCoeff(a::arima, y, nParEst)
    # use OLS results
    T = size(y,1)

    # AR Part
    X = zeros(T-a.p,a.p)
    for i in 1:a.p
      X[:,i] = y[a.p+1-i:end-i]
    end
    pAR = (X'*X)\(X'*y[a.p+1:end])

    # MA part : recover residuals and do OLS
    resid = y[a.p+1:end] - X*pAR
    T = size(resid,1)
    X = zeros(T-a.q,a.q)
    for i in 1:a.q
      X[:,i] = resid[a.q+1-i:end-i]
    end

    pMA = (X'*X)\(X'*resid[a.q+1:end])

    pσ2 = var(resid)

    return  [pAR[isnan.(a.ϕ)]; pMA[isnan.(a.θ)]; isnan.(a.σ2)[1] ? pσ2 : []]
end

function estimate(a::arima, y)
   # difference y by a.d
   for i in 1:a.d
      y = diff(y, dims = 1)
   end

   _estimate(a, y)
end


# end
