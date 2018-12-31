using Revise
using StateSpaceModeling
using BenchmarkTools
using Plots
using CSV

a = arma([.5, -.3],[0.2],.1)
y = simulate(a,1000)
res = estimate(a::arma,y)
