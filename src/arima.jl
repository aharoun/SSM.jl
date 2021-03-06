
# Univariate ARIMA(p,d,q)
mutable struct arima{T<:Real} <: AbstractTimeModel
    p :: Int64		# AR length
    d :: Int64		# Integration order
    q :: Int64		# MA length
    ϕ :: Array{T,1}	# AR coefficients
    θ :: Array{T,1}	# MA coefficients
    σ²:: Array{T,1}	# variance of error term
    c :: Array{T,1}	# constant term

    estimableParamField :: NTuple{4, Symbol}  # holds the fields of a model type that can be estimated

end

# Initialize arima object with empty parameters
function arima(p::Int64,d::Int64,q::Int64)
    ϕ  = fill(NaN,p)
    θ  = fill(NaN,q)
    σ² = fill(NaN,1)
     c = fill(NaN,1)
    estimableParamField = (:ϕ, :θ, :σ², :c)

    arima(p, d, q, ϕ, θ, σ², c, estimableParamField)
end


# Initialize with parameter vectors, some parameters can be set as NaN. Those can be estimated.
function arima(; ϕ::Array{T,1} = [NaN], 
	         θ::Array{T,1} = [NaN], 
	         σ²::Array{T,1} = [NaN],
		 c::Array{T,1} = [NaN], 
		 d::Int64 = 0) where T
    p = length(ϕ)
    q = length(θ)

    estimableParamField = (:ϕ, :θ, :σ², :c)
    arima(p, d, q, ϕ, θ, σ², c, estimableParamField)
end

function Base.show(io::IO, a::arima)
    println(io,"ARIMA(",a.p,",",a.d,",",a.q,") Model")
    println(io,"-------------------")
    println(io," AR: ",a.ϕ)
    println(io," MA: ",a.θ)
    println(io," σ²: ",a.σ²)
    println(io,"  c: ",a.c)
end


# Cast arima model into state space
function StateSpace(a::arima)
    m = max(a.p,a.q + 1)

    A = zeros(1)
    B = zeros(1,m + a.d)
    B[1,1:a.d+1] .= 1.0

    C = zeros(m + a.d)
    C[a.d+1]  = a.c[1]

    G = zeros(m + a.d,m + a.d)
    G[1+a.d:a.p+a.d,1 + a.d]   = a.ϕ
    G[1+a.d:m-1+a.d,2+a.d:end] = diagm(0 => ones(m-1))
    for i in 1:a.d
        G[i,i:a.d+1] .= 1.0
    end

    R = zeros(m + a.d,1)
    R[1+a.d:1+a.q+a.d] = [1.0;a.θ]

    H = zeros(1,1)
    S = fill(a.σ²...,1,1)


    # initialize non stationary states at zero with large variance, stationary states at their unconditional mean and
    # covariance
    nG = size(G,1)

    x0 = zeros(nG)
    x0[a.d+1:end] = (I - G[a.d+1:end,a.d+1:end])\C[a.d+1:end]

    RSR = R*S*R'
    P0   = zeros(nG,nG)
    for i in 1:a.d
       P0[i,i] = 1.0e9 
    end
    P0[1+a.d:end, 1+a.d:end] .= solveDiscreteLyapunov(G[1+a.d:end,1+a.d:end], RSR[1+a.d:end,1+a.d:end])

    StateSpace(A, B, C, G, R, H, S, x0, P0, a)

end

function estimate(a::arima, y)
    size(y,2)>1 ? throw("Arima is applicable for only univariate case") : nothing
    y = repeat(y,1,1)   # just to make it Array{,2}
    _estimate(a, y)
end

# Initialize arima coefficients for estimation
function initializeCoeff(a::arima, y, nParEst)
    # use OLS results
    dy = copy(y)
    for i in 1:a.d
       dy = diff(dy, dims=1)
    end

    T = size(dy,1)

    # AR Part
    X = ones(T-a.p,a.p + 1)
    for i in 1:a.p
      X[:,i+1] = dy[a.p+1-i:end-i]
    end
    coeff = (X'*X)\(X'*dy[a.p+1:end])
    
    pc  = coeff[1]
    pAR = coeff[2:end]

    # MA part : recover residuals and do OLS
    resid = dy[a.p+1:end] - X*coeff
    T = size(resid,1)
    X = zeros(T-a.q,a.q)
    for i in 1:a.q
      X[:,i] = resid[a.q+1-i:end-i]
    end

    pMA = (X'*X)\(X'*resid[a.q+1:end])

    pσ² = var(resid)

    parInit::Array{Float64,1} = vcat([pAR[isnan.(a.ϕ)]; pMA[isnan.(a.θ)]; isnan.(a.σ²)[1] ? pσ² : [] ; isnan.(a.c)[1] ? pc : [] ]...)

    return parInit
 end


# ---------------------------------------------------------------------------------------------------------------

# Model selection

function aicbic(a::arima, y)
    max_p = a.p
    max_q = a.q
    aicAll = zeros(max_p + 1, max_q + 1)
    bicAll = similar(aicAll)
    for p in 0:max_p, q in 0:max_q
	_,_,res  = estimate(arima(p, a.d, q), y)
	aicAll[p+1, q+1] = aic(-res.minf, p + q + 1) # including the constant
	bicAll[p+1, q+1] = bic(-res.minf, p + q + 1, size(y, 1)) # including the constant

    end

    aicTable = NamedArray(aicAll)
    setnames!(aicTable , string.(0:max_p),1)
    setnames!(aicTable , string.(0:max_q),2)
    aicTable.dimnames = ("AR", "MA")

    bicTable = NamedArray(bicAll)
    setnames!(bicTable , string.(0:max_p),1)
    setnames!(bicTable , string.(0:max_q),2)
    bicTable.dimnames = ("AR", "MA")

    return aicTable, bicTable

end


# end

