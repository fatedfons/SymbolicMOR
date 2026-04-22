using Test
using SymbolicMOR

@testset "SymbolicMOR.jl" begin
    include("test_quadratize.jl")
    include("test_pod.jl")
    include("test_lorenz.jl")
end
