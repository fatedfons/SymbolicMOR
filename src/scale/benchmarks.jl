# src/scale/benchmarks.jl
#
# Phase 3 - Speedup Benchmarking Utilities

using Distributed
using BenchmarkTools

"""
    benchmark_scaling(f!, u0_ensemble, tspan; core_counts, dt)

Measure wall-clock time for snapshot generation at each core count in
core_counts and return a NamedTuple for plotting.

# Example
    results = benchmark_scaling(lorenz!, u0s, (0.0, 5.0); core_counts=[1,2,4,8])
    # results.cores   -> [1, 2, 4, 8]
    # results.times   -> elapsed seconds at each core count
    # results.speedup -> speedup relative to single core
"""
function benchmark_scaling(f!, u0_ensemble, tspan;
                           core_counts::Vector{Int}=[1,2,4,8],
                           dt::Float64=0.01)
    times    = Float64[]
    speedups = Float64[]

    t_serial = nothing

    for ncores in core_counts
        t = @elapsed begin
            if ncores == 1
                generate_snapshots(f!, u0_ensemble, tspan; dt)
            else
                generate_snapshots_parallel(f!, u0_ensemble, tspan; dt)
            end
        end

        isnothing(t_serial) && (t_serial = t)

        push!(times, t)
        push!(speedups, t_serial / t)

        @info "Cores: $ncores | Time: $(round(t, digits=3)) s | Speedup: $(round(t_serial / t, digits=2)) x"
    end

    return (cores=core_counts, times=times, speedup=speedups)
end