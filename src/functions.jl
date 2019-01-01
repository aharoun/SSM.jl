
#=
 Model representation
 y_{t} = A + B s_{t} + u
 s_{t} = G s_{t-1} + R*ep
 var(u) ~ H
 var(ep) ~ S
=#
mutable struct StateSpace{T<:Real} 
	A :: AbstractArray{T,1}
	B :: AbstractArray{T,2}
	G :: AbstractArray{T,2}
	R :: AbstractArray{T,2}
	H :: AbstractArray{T,2}
	S :: AbstractArray{T,2}
end



function _initializeKF(ssm::StateSpace,y)
  n = size(ssm.G,1)
  s = zeros(n)
  P = zeros(n,n)
  F = similar(ssm.H) 

  return s, P, F
end


# calculate log likelihood of whole data based on Kalman filter
# Y is Txn matrix where T is sample length and n is the number of variables
function logLike_Y(ssm::StateSpace,y)

      T       = size(y,1)
      s, P, F = _initializeKF(ssm,y)
      ylogL   = 0.0
      RSR     = ssm.R*ssm.S*ssm.R'
      y_fore  = similar(ssm.A)
      pred_err= similar(y_fore)

      @inbounds for i in 1:T
	# forecast
	s .= ssm.G * s
	P .= ssm.G * P * ssm.G' + RSR
	F .= ssm.B * P * ssm.B' + ssm.H

	y_fore   .= ssm.A + ssm.B * s
	pred_err .= y[i,:] - y_fore
	try 
	  ylogL    += (-1/2) * (logdet(F) + pred_err'*(F\pred_err))
	catch
	  ylogL    += 1.0e8
	  break
	end
	# update
	s .+=  P * ssm.B' * (F\pred_err)
	P .-=  P * ssm.B' * (F\ssm.B)*P'

      end

      return ylogL
end


abstract type AbstractTimeModel end


# Univariate ARIMA(p,d,q) without any constant for now
mutable struct arima{T<:Real} <: AbstractTimeModel
    p :: Int64
    d :: Int64
    q :: Int64
    ϕ :: Array{T,1}
    θ :: Array{T,1}
   σ2 :: T
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

    ssm = StateSpace(A, B, G, R, H, S)

end


function simulate(a::arima,T::Int64)
    Random.seed!(1234)
    ssm = StateSpace(a)
    if !all(isempty.(_findEstParamIndex(ssm)))
      throw("Some parameters are not defined!")
    end

    TT = Int64(round(T*1.3))
    y = zeros(TT,length(ssm.A))

    s  = zeros(size(ssm.G,1))

    @inbounds for t in 1:TT
	s      .= ssm.G*s + ssm.R*rand(MvNormal(ssm.S))
	y[t,:]  = ssm.A + ssm.B*s 
    end

    y = y[end-T+1:end]

end

function estimate(a::AbstractTimeModel,y)
    ssm  = StateSpace(a)
    indx = _findEstParamIndex(ssm)

    if all(isempty.(indx))
      throw("Nothing to estimate!")
    end
    estFNames = (:A, :B, :G, :R, :H, :S)[.!isempty.(indx)]
    estFIndex = indx[.!isempty.(indx)]
    nParEst   = sum(length.(estFIndex))


    function objFun(x)
      count = 1
      for (i,valF) in enumerate(estFNames)
	for j in eachindex(estFIndex[i])
	  getproperty(ssm,valF)[estFIndex[i][j]] = x[count]
	  count+=1
	end
      end

      # some checks
      ssm.S[1]<0.0 ? 1.0e8 : -logLike_Y(ssm,y)
    end
      
    pInit = initializeCoeff(a,y,nParEst)

    res = optimize(objFun,pInit,Optim.Options(g_tol = 1.0e-8, iterations = 1000, store_trace = false, show_trace = false))

    return res,ssm
end

_findEstParamIndex(s::StateSpace) = [findall(isnan, getfield(s,fn))  for fn in (:A, :B, :G, :R, :H, :S)]


function initializeCoeff(a::AbstractTimeModel,y,nParEst)
  ones(nParEst)*.1
end

function initializeCoeff(a::arima,y,nParEst)

  # use OLS results to initialize MLE estimation
  if a.d == 0
	dy = y
  elseif a.d == 1
	dy = diff(y)
  else
	dy = diff(diff(y))
  end 
  
  T = size(dy,1)
  # AR Part
  X = zeros(T-a.p,a.p)
  for i in 1:a.p
    X[:,i] = dy[a.p+1-i:end-i]
  end
 
  pAR = (X'*X)\(X'*dy[a.p+1:end])
  # MA part
  resid = dy[a.p+1:end] - X*pAR
  T = size(resid,1)
  X = zeros(T-a.q,a.q)
  for i in 1:a.q
    X[:,i] = resid[a.q+1-i:end-i]
  end
  
  pMA = (X'*X)\(X'*resid[a.q+1:end])

  pσ2 = var(resid)

  [pAR[isnan.(a.ϕ)]; pMA[isnan.(a.θ)]; isnan.(a.σ2) ? pσ2 : []]
end
