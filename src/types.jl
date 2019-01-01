


abstract type AbstractTimeModel end

struct ssmGeneric <: AbstractTimeModel end


# '''
#  Model representation
#  y_{t} = A + B s_{t} + u
#  s_{t} = G s_{t-1} + R*ep
#  var(u) ~ H
#  var(ep) ~ S
# '''

mutable struct StateSpace{T<:Real} 
    A :: Array{T,1}
    B :: Array{T,2}
    G :: Array{T,2}
    R :: Array{T,2}
    H :: Array{T,2}
    S :: Array{T,2}

    model :: AbstractTimeModel
end

function StateSpace(A, B, G, R, H, S)
    m = ssmGeneric()
    StateSpace(A, B, G, R, H, S, m)
end

function Base.show(io::IO, m::StateSpace)
    print(io,"State Space Object ")
    all(isempty.(_findEstParamIndex(m))) ? println("(Fully parametrized)") : println("(With missing parameters)")
    println(io,"-------------------")
    println(io," nObsVar  : ", size(m.B,1))
    println(io," nState   : ", size(m.G,1))
    println(io," modelType: ", typeof(m.model))
end


# Univariate ARIMA(p,d,q) without any constant for now
mutable struct arima{T<:Real} <: AbstractTimeModel
    p :: Int64
    d :: Int64
    q :: Int64
    ϕ :: Array{T,1}
    θ :: Array{T,1}
   σ2 :: T
end

function Base.show(io::IO, a::arima)
    println(io,"ARIMA(",a.p,",",a.d,",",a.q,") Model")
    println(io,"-------------------")
    println(io," AR: ",a.ϕ)
    println(io," MA: ",a.θ)
    println(io," σ2: ",a.σ2)
end

