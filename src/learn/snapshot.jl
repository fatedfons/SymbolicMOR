# src/learn/snapshot.jl
#
# Phase 2 - Snapshot Matrix Generation
#
# Simulate the lifted ODE over many initial conditions to build
# the snapshot matrix X in R^(n x K), where n = lifted state dim
# and K = total number of time steps across all trajectories.

using OrdinaryDiffEq
using LinearAlgebra

"""
    generate_snapshots(f!, u0_ensemble, tspan; kwargs...)

Simulate the lifted ODE du/dt = f!(du, u, p, t) from each initial
condition in u0_ensemble and collect all state snapshots into a matrix.

# Arguments
- f!            : in-place ODE function f!(du, u, p, t)
- u0_ensemble   : vector of initial conditions (each a Vector{Float64})
- tspan         : (t_start, t_end) tuple
- dt            : time step for saved snapshots (default: 0.01)
- solver        : ODE solver (default: Tsit5())

# Returns
X::Matrix{Float64} of shape (state_dim, n_snapshots_total)

# Example
```julia
function lorenz!(du, u, p, t)
    sigma, rho, beta = 10.0, 28.0, 8/3
    du[1] = sigma * (u[2] - u[1])
    du[2] = u[1] * (rho - u[3]) - u[2]
    du[3] = u[1] * u[2] - beta * u[3]
end

u0s = [randn(3) .+ [1.0, 0.0, 25.0] for _ in 1:20]
X = generate_snapshots(lorenz!, u0s, (0.0, 10.0))
```
"""
function generate_snapshots(
    f!,
    u0_ensemble::Vector{<:AbstractVector},
    tspan::Tuple{<:Real,<:Real};
    dt::Float64 = 0.01,
    solver = Tsit5(),
    kwargs...
)
    all_snapshots = Vector{Matrix{Float64}}()

    for u0 in u0_ensemble
        prob = ODEProblem(f!, u0, tspan)
        sol  = solve(prob, solver; saveat=dt, kwargs...)
        # Each column is one time snapshot
        push!(all_snapshots, hcat(sol.u...))
    end

    return hcat(all_snapshots...)
end
