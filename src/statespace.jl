
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


"""
    estimate(s:StateSpace,y)
Estimates state space model. All the entries of the state space matrices with NaN are considered as unknown parameters to be estimated.

"""
function estimate(ssm::StateSpace,y)

    estFNames, estFIndex, nParEst = getEstParamIndexSSM(ssm::StateSpace)

    pInit = initializeCoeff(ssm.model,y,nParEst)

    res   = optimize(x -> ssmNegLogLike!(x, ssm, y, estFNames, estFIndex),
		      pInit,
		      Optim.Options(g_tol = 1.0e-8, iterations = 1000, store_trace = false, show_trace = false))

    ssmNegLogLike!(res.minimizer, ssm, y, estFNames, estFIndex)
    return ssm,res
end

function estimate(a::AbstractTimeModel,y)
    ssm     = StateSpace(a)
    ssm,res = estimate(ssm::StateSpace,y)
    getParamFromSSM!(ssm,a)
    return a, res
end


function getEstParamIndexSSM(ssm::StateSpace)
    # mask parameters to be estimated
    indx = _findEstParamIndex(ssm)

    if all(isempty.(indx))
	throw("Nothing to estimate!")
    end
    estFNames = (:A, :B, :G, :R, :H, :S)[.!isempty.(indx)]
    estFIndex = indx[.!isempty.(indx)]
    nParEst   = sum(length.(estFIndex))

    return estFNames, estFIndex, nParEst
end


function getParamFromSSM!(ssm::StateSpace, a::arima)
     a.ϕ .= ssm.G[1+a.d:a.p+a.d,1 + a.d]
     a.θ .= ssm.R[2+a.d:1+a.q+a.d]
     a.σ2 = ssm.S[1]
end

# objective function for `estimate`
function ssmNegLogLike!(x,ssm::StateSpace, y, estFNames, estFIndex)
    count = 1
    @inbounds for (i,valF) in enumerate(estFNames)
      for j in eachindex(estFIndex[i])
	getproperty(ssm,valF)[estFIndex[i][j]] = x[count]
	count+=1
      end
    end

    # more checks needed
    ssm.S[1]<0.0 ? 1.0e8 : -logLike_Y(ssm,y)
end


#--------------------------------------------------------------------------------------------------------------
#
_findEstParamIndex(s::StateSpace) = [findall(isnan, getfield(s,fn))  for fn in (:A, :B, :G, :R, :H, :S)]


function initializeCoeff(a::AbstractTimeModel, y, nParEst)
    pInit = ones(nParEst)*0.1
end

