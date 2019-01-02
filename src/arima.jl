
# Univariate ARIMA(p,d,q) without any constant for now
mutable struct arima{T<:Real} <: AbstractTimeModel
    p :: Int64		# AR length
    d :: Int64		# Integration order
    q :: Int64		# MA length
    ϕ :: Array{T,1}	# AR coefficients
    θ :: Array{T,1}	# MA coefficients
   σ2 :: T		# variance of error term
end

# Initialize arima object with empty parameters
function arima(p::Int64,d::Int64,q::Int64)
    ϕ  = fill(NaN,p)
    θ  = fill(NaN,q)
    σ2 = NaN

    arima(p,d,q,ϕ,θ,σ2)
end

function arima(ϕ::Array{T,1}, θ::Array{T,1}, σ2::T, d::Int64) where T
    p = length(ϕ)
    q = length(θ)

    arima(p,d,q,ϕ,θ,σ2)
end

function Base.show(io::IO, a::arima)
    println(io,"ARIMA(",a.p,",",a.d,",",a.q,") Model")
    println(io,"-------------------")
    println(io," AR: ",a.ϕ)
    println(io," MA: ",a.θ)
    println(io," σ2: ",a.σ2)
end


# Cast arima model into state space
function StateSpace(a::arima)
    m = max(a.p,a.q + 1)

    A = zeros(1)
    B = zeros(1,m + a.d)
    B[1:a.d+1] .= 1.0

    G = zeros(m + a.d,m + a.d)
    G[1+a.d:a.p+a.d,1 + a.d]   = a.ϕ
    G[1+a.d:m-1+a.d,2+a.d:end] = diagm(0 => ones(m-1))
    for i in 1:a.d
	G[i,i:a.d+1] .= 1.0
    end

    R = zeros(m + a.d,1)
    R[1+a.d:1+a.q+a.d] = [1.0;a.θ]

    H = zeros(1,1)
    S = fill(a.σ2,1,1)

    StateSpace(A, B, G, R, H, S, a)

end

# simulate arima model. This can be done generic but MvNormal does not like zero variance
function simulate(a::arima,T::Int64)
    Random.seed!(20)
    ssm = StateSpace(a)
    if !all(isempty.(findEstParamIndex(ssm)))
      throw("Some parameters are not defined!")
    end

    TT = Int64(round(T*1.5))
    y = zeros(TT,length(ssm.A))

    s  = zeros(size(ssm.G,1))

    @inbounds for t in 1:TT
	s      .= ssm.G*s + ssm.R*rand(MvNormal(ssm.S))
	y[t,:]  = ssm.A + ssm.B*s 
    end
    # Discard initial draws
    y = y[end-T+1:end]

end

# Initialize arima coefficients for estimation
function initializeCoeff(a::arima, y, nParEst)
    # use OLS results
    dy = y
    for i in 1:a.d
	dy = diff(dy)
    end

    T = size(dy,1)

    # AR Part
    X = zeros(T-a.p,a.p)
    for i in 1:a.p
      X[:,i] = dy[a.p+1-i:end-i]
    end
    pAR = (X'*X)\(X'*dy[a.p+1:end])

    # MA part : recover residuals and do OLS
    resid = dy[a.p+1:end] - X*pAR
    T = size(resid,1)
    X = zeros(T-a.q,a.q)
    for i in 1:a.q
      X[:,i] = resid[a.q+1-i:end-i]
    end

    pMA = (X'*X)\(X'*resid[a.q+1:end])

    pσ2 = var(resid)

    return  [pAR[isnan.(a.ϕ)]; pMA[isnan.(a.θ)]; isnan.(a.σ2) ? pσ2 : []]
end


function getParamFromSSM!(ssm::StateSpace, a::arima)
     a.ϕ .= ssm.G[1+a.d:a.p+a.d,1 + a.d]
     a.θ .= ssm.R[2+a.d:1+a.q+a.d]
     a.σ2 = ssm.S[1]
end

# end
