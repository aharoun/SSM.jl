
abstract type AbstractTimeModel end

struct ssmGeneric <: AbstractTimeModel end



#  Model representation
#  y_{t} = A + B s_{t} + u
#  s_{t} = G s_{t-1} + R*ep
#  var(u) ~ H
#  var(ep) ~ S

struct StateSpace{T<:Real} 
    A :: Array{T,1}
    B :: Array{T,2}
    G :: Array{T,2}
    R :: Array{T,2}
    H :: Array{T,2}
    S :: Array{T,2}

    model :: AbstractTimeModel
end

# initialize a generic state space model
function StateSpace(A, B, G, R, H, S)
    m = ssmGeneric()
    StateSpace(A, B, G, R, H, S, m)
end

function Base.show(io::IO, m::StateSpace)
    println(io, "State Space Object for: ", m.model)
end

# -----------------------------------------------------------------------------------------------

# calculate log likelihood of whole data based on Kalman filter
# Y is Txn matrix where T is sample length and n is the number of variables
function nLogLike(ssm::StateSpace, y)

    T       = size(y,1)
    s, P, F = _initializeKF(ssm,y)
    ylogL   = zeros(T)
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
	    ylogL[i]= (-1/2) * (logdet(F) + pred_err'*(F\pred_err))
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


# TODO : initialize P based on unconditional distribution  
function _initializeKF(ssm::StateSpace,y)
    n = size(ssm.G,1)
    s = zeros(n)
    P = zeros(n,n)
    F = similar(ssm.H) 

    return s, P, F
end

# -----------------------------------------------------------------------------------------------

"""
    _estimate(s:AbstractTimeModel,y)
Estimates time series model. All the entries of the estimable parameters with NaN are considered as unknown parameters to be estimated.

"""
function _estimate(a::AbstractTimeModel, y)
    a = deepcopy(a)

    estPIndex = findEstParamIndex(a)
    nParEst   = length(vcat(estPIndex...))

    pInit = initializeCoeff(a, y, nParEst)

    objFun = x -> sum(negLogLike!(x, a, y, estPIndex))
    res    = optimize(objFun,
		      pInit,
		      Optim.Options(g_tol = 1.0e-12, iterations = 1000, store_trace = false, show_trace = false))

    stdErr = stdErrParam(res.minimizer, x -> negLogLike!(x, a, y, estPIndex))

    negLogLike!(res.minimizer, a, y, estPIndex)    # to cast ssm at minimizer

    return a, res, stdErr
end

function estimate(a::AbstractTimeModel, y)
    _estimate(a, y)
end

# objective function for `estimate`
function negLogLike!(x, a::AbstractTimeModel, y, estPIndex)
    setEstParam!(x, a::AbstractTimeModel, estPIndex)
    ssm = StateSpace(a)
    # more checks needed
    ssm.S[1]<0.0 ? 1.0e8 : -nLogLike(ssm, y)
end

function setEstParam!(x, a, estPIndex)
    count = 1
    @inbounds for (i, valF) in enumerate(a.estPar)
		  for j in estPIndex[i]
		      getproperty(a, valF)[j] = x[count]
			  count+=1
	          end
	      end
end

# Standard error of the estimated parameters, based on outer product of score
function stdErrParam(parEst,nlogl::Function)
    score  = Calculus.jacobian(nlogl, parEst, :central)
    cov = pinv(score'*score)
    cov = (cov + cov')/2
    std = sqrt.(diag(cov))
end

# -----------------------------------------------------------------------------------------------

# Utility function

function initializeCoeff(a::AbstractTimeModel, y, nParEst)
    pInit = ones(nParEst)*0.1
end


findEstParamIndex(a::AbstractTimeModel) = [findall(isnan, getfield(a,fn))  for fn in a.estPar]


# end
