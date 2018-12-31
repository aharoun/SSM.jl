
#=
 Model representation
 y_{t} = A + B s_{t} + u
 s_{t} = Phi s_{t-1} + R*ep
 var(u) ~ H
 var(ep) ~ S
=#

mutable struct StateSpaceModel{Tv<:AbstractArray{Float64,1},Tm<:AbstractArray{Float64,2}}
	A :: Tv
	B :: Tm
	Φ :: Tm
	R :: Tm
	H :: Tm
	S :: Tm

end



function initialize(ssm::StateSpaceModel)
  n = size(ssm.Φ,1)
  s = zeros(n)
  P = zeros(n,n)
  F = similar(ssm.H) 

  return s, P, F
end


# calculate log likelihood of whole data based on Kalman filter
# Y is Txn matrix where T is sample length and n is the number of variables
function logLike_Y(ssm::StateSpaceModel,y)

      T       = size(y,1)
      s, P, F = initialize(ssm)
      ylogL   = 0.0
      RSR     = ssm.R*ssm.S*ssm.R'
      y_fore  = similar(ssm.A)
      pred_err= similar(y_fore)

      @inbounds for i in 1:T
	# forecast
	s .= ssm.Φ * s
	P .= ssm.Φ * P * ssm.Φ' + RSR
	F .= ssm.B * P * ssm.B' + ssm.H

	y_fore   .= ssm.A + ssm.B * s
	pred_err .= y[i,:] - y_fore
	ylogL    += (-1/2) * (logdet(F) + pred_err'*(F\pred_err))

	# update
	s .+=  P * ssm.B' * (F\pred_err)
	P .-=  P * ssm.B' * (F\ssm.B)*P'

      end

      return ylogL
end


# Univariate Arma(p,q)
mutable struct arma

    p :: Int64
    q :: Int64
    ϕ :: Array{Float64,1}
    θ :: Array{Float64,1}
   σ2 :: Float64
  ssm :: StateSpaceModel

end

function arma(ϕ::Array{Float64,1},θ::Array{Float64,1},σ2::Float64)
    p = length(ϕ)
    q = length(θ)

    m = max(p,q + 1)

    # write state space version as arma(m,m-1)
    A = zeros(1)
    B = zeros(1,m)
    B[1] = 1.0

    Φ = zeros(m,m)
    Φ[1:p] = ϕ
    Φ[1:m-1,2:end] = (Matrix(1.0I, m-1, m-1))
 
    R = ones(m,1)
    R[2:q+1] = θ

    H = zeros(1,1)

    S = fill(σ2,1,1)

    ssm = StateSpaceModel(A, B, Φ, R, H, S)

    arma(p, q, ϕ, θ, σ2, ssm)
end

arma(ϕ,θ) = arma(ϕ,θ,1.0) 

function arma(p::Int64,q::Int64)
    ϕ = zeros(p)
    θ = zeros(q)

    arma(ϕ,θ)

end


function simulate(a::arma,T::Int64)

    ssm = a.ssm

    TT = Int64(round(T*1.3))
    y = zeros(TT,length(ssm.A))

    s  = zeros(size(ssm.Φ,1))

    @inbounds for t in 1:TT
	s      .= ssm.Φ*s + ssm.R*rand(MvNormal(ssm.S))
	y[t,:]  = ssm.A + ssm.B*s 
    end

    y = y[end-T+1:end]

end

function estimate(a::arma,y)
    p = a.p
    q = a.q

    pInit  = [ones(p)*.1;ones(q)*.1;.1]

    objFun(x) = x[end]<=0 ? 1.0e8 : -logLike_Y(arma(x[1:p],x[p+1:p+q],x[end]).ssm,y)

    res = optimize(objFun,pInit,show_trace=false)
end

######################################################################################

function modelTechAdpotion(x,nLambda,n)
    W1 = zeros(2,2);
    W1[1,1] = x[1]; W1[1,2] = x[2]; W1[2,1] = x[2]; W1[2,2] = x[3];
    #------------------------------------------------------------

    #------------------------------------------------------------
   #State space VAR coefficients
    G1 = zeros((nLambda+2),(nLambda+2))

    G1[1:2,1:2] = reshape(x[4:7],2,2);
    G1[3:(nLambda+2),2:(nLambda+1)] = diagm(0=>ones(nLambda))
   #------------------------------------------------------------

   #------------------------------------------------------------
   #Observation equation var-cov matrix
   V1 = diagm(0=>x[8:7+n])
   #------------------------------------------------------------

   #------------------------------------------------------------
   #Selection matrix
    R = zeros((nLambda+2), 2);# Selection matrix, not cov matrix (kalman filter reverted to its original version acccordingy)
    R[1:2,1:2] = diagm(0 => ones(2))
   #------------------------------------------------------------

    F1 = zeros(n,nLambda+2);
   #------------------------------------------------------------
   #Non-tech and tech lag contributions matrix
   M1 = reshape(x[7+n+1:7+ n + 3*n],3,n)'

    for i in  1:n
	for j in 2:(nLambda+2)
	  F1[i,j] = M1[i,1]*exp(-((((j-1) - M1[i,2])^2)/M1[i,3]))
	end
    end
   #------------------------------------------------------------

   #------------------------------------------------------------
   #Contemparenous non-tech and tech contributions
   F1[:,1] = x[7+ n + 3*n+1:7+ n + 3*n+n]
   #------------------------------------------------------------


   # kalman filter form
    A = zeros(n);# this is constant in y-s equation.
    B = F1
    H = V1
    Φ = G1
    S = W1

    ssm = StateSpaceModel(A, B, Φ, R, H, S)
end
