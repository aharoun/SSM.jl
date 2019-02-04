
abstract type AbstractTimeModel end

struct ssmGeneric <: AbstractTimeModel end

# State Space Model representation
# y(t) = A + B×s(t) + u
# s(t) = C + G×s(t-1) + R×ep
# u  ~ N(0,H)
# ep ~ N(0,S)
# s(0) ~ N(x0,P0)

# TODO: dimension checking
struct StateSpace{T<:Real} 
    A :: Array{T,1}
    B :: Array{T,2}
    C :: Array{T,1}
    G :: Array{T,2}
    R :: Array{T,2}
    H :: Array{T,2}
    S :: Array{T,2}
    x0:: Array{T,1}
    P0:: Array{T,2} 

    model :: AbstractTimeModel

end

# initialize a generic state space model and calculate x0 and P0 based on unconditional distribution
function StateSpace(A, B, C, G, R, H, S)

    n = size(G,1)

    if maximum(abs.(eigvals(G)))>=0.9999
	warning("State space representation is nonstationary. All states will be initialized in a diffuse way.")
	x0 = zeros(n)
	P0 = diagm(0 => ones(n) * 1.0e8)
    else
	s0 = (I - ssm.G)\ssm.C
	P0 = solveDiscreteLyapunov(ssm.G, ssm.R*ssm.S*ssm.R')
    end

    m = ssmGeneric()
    StateSpace(A, B, C, G, R, H, S, x0, P0, m)
end

function Base.show(io::IO, m::StateSpace)
    println(io, "State Space Object for: ", m.model)
end

# -----------------------------------------------------------------------------------------------

# simulate model
function simulate(ssm::StateSpace, T::Int64)
    TT = Int64(round(T*1.5))
    y = zeros(length(ssm.A),TT)

    nG = size(ssm.G,1)
    nS = size(ssm.S,1)
    nH = size(ssm.H,1)

    s  = zeros(nG)
    cholS = all(ssm.S .== 0.0) ? ssm.S : cholesky(ssm.S).L
    cholH = all(ssm.H .== 0.0) ? ssm.H : cholesky(ssm.H).L

    @inbounds for t in 1:TT
	s      .= ssm.C + ssm.G*s + ssm.R*cholS*randn(nS)
	y[:,t] .= ssm.A + ssm.B*s + cholH*randn(nH)
    end

    y = y'
    # Discard initial draws
    y = y[end-T+1:end,:]

end

function simulate(a::AbstractTimeModel, T)
    ssm = StateSpace(a)
    if !all(isempty.(findEstParamIndex(a)))
      throw("Some parameters are not defined!")
    end

    simulate(ssm, T)
end
#--------------------------------------------------------------------------------------------------------------
# calculate log likelihood of whole data based on Kalman filter
# Y is Txn matrix where T is sample length and n is the number of variables
# TODO: missing data support (it is very easy) 
function nLogLike(a::AbstractTimeModel, y)

    ssm = StateSpace(a)

    T       = size(y,1)
    s       = copy(ssm.x0)
    P       = copy(ssm.P0)
    sF      = similar(s)
    PF      = similar(P)
    copyP   = similar(P)
    F       = similar(ssm.H)
    ylogL   = zeros(T)
    RSR     = ssm.R*ssm.S*ssm.R'
    y_fore  = similar(ssm.A)
    pred_err= similar(y_fore)

    converged = false
    i = 1

    @inbounds while i<=T && !converged
        # forecast
        sF .= ssm.C .+ ssm.G * s
        PF.= ssm.G * P * ssm.G' .+ RSR
        
        F .= ssm.B * PF * ssm.B' .+ ssm.H

        y_fore   .= ssm.A  .+ ssm.B * sF
        pred_err .= y[i,:] .- y_fore
        try
            ylogL[i] = (-1/2) * (logdet(F) + pred_err'*(F\pred_err))
        catch
            ylogL[i] = -Inf
        end
        # update
        s .= sF .+ PF * ssm.B' * (F\pred_err)
        P .= PF .- PF * ssm.B' * (F\ssm.B)*PF'
        
        # break if P, PF and F are converged, go to the second stage
        maximum(abs.(P .- copyP))<1.0e-18 ? converged = true : nothing
        
        copyP = copy(P)

        i += 1
    end

    # taking advantage of the fact that P and F are converged so that we don't need to update them
    @inbounds for j in i:T
        # forecast
	sF       .= ssm.C .+ ssm.G * s
	y_fore   .= ssm.A  .+ ssm.B * sF
	
	pred_err .= y[j,:] .- y_fore

	try
	    ylogL[j] = (-1/2) * (logdet(F) + pred_err'*(F\pred_err)) 
	catch
	    ylogL[j] = -Inf
	end
	## update
	s .= sF .+ PF * ssm.B' * (F\pred_err)	
    end

    return ylogL .- 0.5*log(2*pi)
 
end

# -----------------------------------------------------------------------------------------------

"""
    _estimate(s:AbstractTimeModel,y)
Estimates time series model. All the entries of the estimable parameters with NaN are considered as unknown parameters to be estimated.

"""
function _estimate(a::AbstractTimeModel, y)
    estPIndex = findEstParamIndex(a)
    nParEst   = length(vcat(estPIndex...))
    pInit     = initializeCoeff(a, y, nParEst)
   

    nParEst != length(pInit) ? throw("Initial number of parameter is not consistent with the number of parameters to be estimated!") : nothing

    res = optimize(x -> sum(negLogLike!(x, a, y, estPIndex)),
		   pInit,
		   Optim.Options(g_tol = 1.0e-8, iterations = 1000, store_trace = false, show_trace = false))

    stdErr = stdErrParam(res.minimizer, x -> negLogLike!(x, a, y, estPIndex))

    negLogLike!(res.minimizer, a, y, estPIndex)    # to cast the model parameters at minimizer
    
    resTable = NamedArray([res.minimizer stdErr])
    setnames!(resTable, ["Point Est.", "Std. Error"], 2)
    resTable.dimnames = ("Parameters", "")
    return a, resTable, res
end

function estimate(a::AbstractTimeModel, y)
    _estimate(a, y)
end

# objective function for `estimate`
function negLogLike!(x, a::AbstractTimeModel, y, estPIndex)
    setEstParam!(x, a, estPIndex)
    # more checks needed
    -nLogLike(a, y)
end

function setEstParam!(x, a, estPIndex)
    count = 1
    @inbounds for (i, valF) in enumerate(a.estimableParamField)
		  for j in estPIndex[i]
		      getproperty(a, valF)[j] = x[count]
		      count += 1
	          end
	      end
end

# Standard error of the estimated parameters, based on outer product of score
function stdErrParam(parEst,nlogl::Function)
    score  = Calculus.jacobian(nlogl, parEst, :central)
    cov    = pinv(score'*score)
    cov    = (cov + cov')/2
    std    = sqrt.(diag(cov))
end



function forecast(a::AbstractTimeModel, y, Tf)

    ssm = StateSpace(a)

    T       = size(y,1)
    s       = ssm.x0
    P       = ssm.P0
    F       = similar(ssm.H)
    ylogL   = zeros(T)
    RSR     = ssm.R*ssm.S*ssm.R'
    y_fore  = similar(ssm.A)
    pred_err= similar(y_fore)

    yForecast = zeros(size(y,2),Tf)
    FForecast = zeros(size(y,2),Tf)

    @inbounds for i in 1:T
	# forecast
	s .= ssm.C .+ ssm.G * s
	P .= ssm.G * P * ssm.G' .+ RSR
	F .= ssm.B * P * ssm.B' .+ ssm.H

	y_fore   .= ssm.A  .+ ssm.B * s
	pred_err .= y[i,:] .- y_fore

	# update
	s .+=  P * ssm.B' * (F\pred_err)
	P .-=  P * ssm.B' * (F\ssm.B)*P'
    end

    for i in 1:Tf
	s .= ssm.C .+ ssm.G * s
	P .= ssm.G * P * ssm.G' .+ RSR
	
	yForecast[:,i] .= ssm.A .+ ssm.B * s
	FForecast[:,i] .= diag(ssm.B * P * ssm.B' .+ ssm.H)
	#  no update
    end

    return yForecast', FForecast'
end


# -----------------------------------------------------------------------------------------------

# Utility function

function initializeCoeff(a::AbstractTimeModel, y, nParEst)
    pInit = ones(nParEst)*0.1
end


findEstParamIndex(a::AbstractTimeModel)::Array{Array{Int64,1},1} = [findall(isnan, getfield(a,fn))  for fn in a.estimableParamField]

# Solve discrete Lyapunov equation AXA' - X + B = 0.
function solveDiscreteLyapunov(A::Array{T,2}, B::Array{T,2}) where T
    aux = I - kron(A, A)
    sol = aux\vec(B)
    return reshape(sol,size(B))
end

# make deep copy type stable
Base.deepcopy(m::MyType) where MyType <: AbstractTimeModel = begin
    len = length(fieldnames(typeof(m)))
    box = Array{Any,1}(undef,len)
    
    for i in 1:len

	try 
	    box[i] = copy(getfield(m, fieldnames(typeof(m))[i]))
	catch
	    box[i] = getfield(m, fieldnames(typeof(m))[i])
	end
    end
	MyType(box...)
end
# end
