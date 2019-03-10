
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
    ϕ::Array{Real,1}   = fill(NaN,p)
    θ::Array{Real,1}   = fill(NaN,q)
    σ²::Array{Real,1}  = fill(NaN,1)
    c::Array{Real,1}   = fill(NaN,1)
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

    A = zeros(eltype(a.c[1]),1)
    B = zeros(eltype(a.c[1]),1,m + a.d)
    B[1,1:a.d+1] .= 1.0

    C = zeros(eltype(a.c[1]),m + a.d)
    C[a.d+1]  = a.c[1]

    G = zeros(eltype(a.c[1]),m + a.d,m + a.d)
    G[1+a.d:a.p+a.d,1 + a.d]   = a.ϕ
    G[1+a.d:m-1+a.d,2+a.d:end] = diagm(0 => ones(m-1))
    for i in 1:a.d
        G[i,i:a.d+1] .= 1.0
    end

    R = zeros(eltype(a.c[1]),m + a.d,1)
    R[1+a.d:1+a.q+a.d] = [1.0;a.θ]

    H = zeros(eltype(a.c[1]),1,1)
    S = fill(a.σ²...,1,1)


    # initialize non stationary states at zero with large variance, stationary states at their unconditional mean and
    # covariance
    nG = size(G,1)

    x0 = zeros(eltype(a.c[1]),nG)
    # x0[a.d+1:end] = (I - G[a.d+1:end,a.d+1:end])\C[a.d+1:end]

    RSR = R*S*R'
    P0   = zeros(eltype(a.c[1]),nG,nG)
    for i in 1:a.d
       P0[i,i] = 1.0e9 
    end
    # P0[1+a.d:end, 1+a.d:end] .= solveDiscreteLyapunov(G[1+a.d:end,1+a.d:end], RSR[1+a.d:end,1+a.d:end])
    
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


function initializeCoeff2(a::arima, y, nParEst)
    dy = copy(y)
    for i in 1:a.d
	dy = diff(dy, dims=1)
    end

    T = size(dy,1)
    meanDy = mean(dy)
    dy .-= meanDy  #demean
    
    # autocovariance
    cov = zeros(a.p + a.q+ 1)
    for i in 0:(a.p + a.q) 
	cov[i+1] = dot(dy[1:end-i],dy[1+i:end])/T
    end

    # solve AR coeff
    A = zeros(a.p, a.p)
    for i in 1:a.p, j in 1:a.p
	A[i,j] = cov[abs(a.q + i - j) + 1]
    end
    b = cov[a.q .+ (1:a.p) .+ 1]
    
    ϕInit = A\b
    if abs(sum(ϕInit))>1.0
	ϕInit = zeros(length(ϕInit))    
    end

    #Solve MA
    covM = zeros(a.q + 1) # modified autcovariance
    ϕInitA = [-1.0; ϕInit]
   
    if a.p == 0
	covM .= cov[1:a.q+1]
    else
	for j in 0:a.q
	    aux = 0.0
	    for i in 0:a.p
		for k in 0:a.p
		    aux += ϕInitA[i+1]*ϕInitA[k+1]*cov[abs(i + j - k) + 1]
		end
	    end
	    covM[j+1] = aux
	end
    end

    # iteration
    crit = 1.0; iter = 0
    θ = zeros(a.q+1)
    θCopy = ones(a.q+1)
    σ² = 0.1

    while crit>1.0e-8 && iter<100
	σ² = covM[1]/(1 + sum(θ[2:end].^2))

	for j in a.q:-1:1
	    aux1 = zeros(a.q-1)
	    aux2 = zeros(a.q-1)

	    aux1[1:a.q-j] .= θ[2:a.q-j+1]
	    aux2[1:a.q-j] .= θ[j+2:a.q+1]
	    θ[j+1] = covM[j+1]/σ² - sum(aux1.*aux2)
	end
	crit = maximum(abs, θ .- θCopy)
	θCopy = copy(θ)
	iter += 1 
    end
    if isnan(crit) || isinf(crit) || crit>1.0e-8 || iter>=100
	θ  = ones(a.q)*0.1
	σ² = 0.1
    else
	θ = θ[2:end]
    end

    cInit = a.p==0 ? meanDy : meanDy*(1.0 - sum(ϕInit))
    parInit::Array{Float64,1} = vcat([ϕInit[isnan.(a.ϕ)]; θ[isnan.(a.θ)]; isnan.(a.σ²)[1] ? σ² : [] ; isnan.(a.c)[1] ? cInit : [] ]...)


    # Newton-Raphson
    # τ =  zeros(a.q+1)
    # τ[1] = sqrt(covM[1])
    # τNew = similar(τ)

    # crit = 1.0
    # maxIter = 1000
    # iter = 0
    # f = ones(a.q+1)
    # TT1 = zeros(a.q+1, a.q+1)
    # TT2 = similar(TT1)
    # TT  = similar(TT1)

    # while crit>1.0e-8 && iter<maxIter
	# for j in 0:a.q
	    # f[j+1] = dot(τ[1:a.q - j + 1], τ[1+j:end]) - covM[j+1]
	# end
	
	# for i in 1:a.q+1
	    # TT1[i,1:end-i+1] = τ[i:end] 
	    # TT2[i,i:end] = τ[1:end-i+1]
	# end
	# TT .= TT1 .+ TT2

	# h = TT\f
	# τNew .= τ .- h 

	# crit = maximum(abs.(τ.- τNew))
	# τ .= τNew 
	# iter += 1
    # end
    
    # if crit>1.0e-8
	# println("Didn't converge!!!")
	# θInit = zeros(length(τ)-1)
    # else
	# θInit = τ[2:end]./τ[1]
    # end
    # σ²Init = τ[1]^2

end


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

