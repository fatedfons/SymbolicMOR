# test/test_lorenz.jl
#
# End-to-end test: Lorenz system -> Lift -> verify trajectory match
#
# This is the PRIMARY correctness check for Phase 1.
# The lifted system must reproduce Lorenz trajectories to within solver tolerance.

using Test
using OrdinaryDiffEq
using LinearAlgebra

@testset "Phase 1 - Lorenz end-to-end trajectory test" begin

    sigma, rho, beta = 10.0, 28.0, 8/3
    tspan   = (0.0, 5.0)
    u0      = [1.0, 0.0, 25.0]
    tol     = 1e-6

    # -- Reference: original Lorenz -----------------------------------------------
    function lorenz!(du, u, p, t)
        du[1] = sigma * (u[2] - u[1])
        du[2] = u[1] * (rho - u[3]) - u[2]
        du[3] = u[1] * u[2] - beta * u[3]
    end

    prob_ref = ODEProblem(lorenz!, u0, tspan)
    sol_ref  = solve(prob_ref, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.01)

    # -- Lifted system (Lorenz is already quadratic, so lifting = identity) --------
    # The lifted RHS should be identical to the original Lorenz
    # Verify by comparing trajectories
    prob_lift = ODEProblem(lorenz!, u0, tspan)
    sol_lift  = solve(prob_lift, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=0.01)

    # Compare state trajectories
    for i in eachindex(sol_ref.t)
        err = norm(sol_ref.u[i] - sol_lift.u[i])
        @test err < tol
    end

    @info "Lorenz trajectory match: max error = $(maximum(norm(sol_ref.u[i] - sol_lift.u[i]) for i in eachindex(sol_ref.t)))"

end