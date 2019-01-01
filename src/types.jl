


abstract type AbstractTimeModel end

struct ssmGeneric <: AbstractTimeModel end
'''
 Model representation
 y_{t} = A + B s_{t} + u
 s_{t} = G s_{t-1} + R*ep
 var(u) ~ H
 var(ep) ~ S
'''

mutable struct StateSpace{T<:Real} 
    A :: Array{T,1}
    B :: Array{T,2}
    G :: Array{T,2}
    R :: Array{T,2}
    H :: Array{T,2}
    S :: Array{T,2}

    model :: AbstractTimeModel
end

function StateSpace(A, B, R, H, S)
  m = ssmGeneric()
  StateSpace(A, B, R, H, S, m)
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

