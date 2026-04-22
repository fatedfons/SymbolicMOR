# src/learn/pod_galerkin.jl
#
# Phase 2 - POD + Galerkin Projection
#
# Given the snapshot matrix X, compute a low-dimensional basis via
# Proper Orthogonal Decomposition (POD), then project the lifted
# quadratic ODE onto this basis to obtain the Reduced-Order Model (ROM).

using LinearAlgebra

"""
    compute_pod_basis(X, r; energy_threshold=0.9999)

Compute a POD basis from the snapshot matrix X in R^(n x K).

# Arguments
- X                : snapshot matrix (rows = state dims, cols = snapshots)
- r                : number of POD modes to retain (pass `nothing` to use threshold)
- energy_threshold : if r is `nothing`, retain enough modes to capture this
                     fraction of total energy (default: 99.99%)

# Returns
(Phi, sigma, energy) where:
  - Phi::Matrix{Float64}   : POD basis, shape (n, r), orthonormal columns
  - sigma::Vector{Float64} : singular values
  - energy::Float64        : fraction of energy captured by chosen modes
"""
function compute_pod_basis(X::Matrix{Float64}, r::Union{Int,Nothing}=nothing;
                           energy_threshold::Float64=0.9999)
    U, sigma, _ = svd(X)

    # Determine r from energy threshold if not specified
    if isnothing(r)
        total_energy = sum(sigma.^2)
        cumulative   = cumsum(sigma.^2) ./ total_energy
        r = findfirst(>=(energy_threshold), cumulative)
        isnothing(r) && (r = length(sigma))
        @info "POD: retaining r=$r modes ($(round(cumulative[r]*100, digits=4))% energy)"
    end

    r = min(r, size(U, 2))
    Phi = U[:, 1:r]
    energy = sum(sigma[1:r].^2) / sum(sigma.^2)

    return Phi, sigma, energy
end

"""
    galerkin_project(A, H, c, Phi)

Project the lifted quadratic RHS onto the POD basis Phi to produce the ROM.

For a lifted system  s_dot = F(s)  (quadratic), let  s approx Phi a,
then the ROM is:
    a_dot = Phi' * F(Phi * a)

In matrix form for a quadratic system
    F(s) = A*s + H*(s kron s) + c

the ROM operators are:
    A_hat = Phi' * A * Phi
    H_hat = Phi' * H * (Phi kron Phi)
    c_hat = Phi' * c

# Arguments
- A   : linear operator matrix (n x n)
- H   : quadratic operator (n x n^2)
- c   : constant term (n,)
- Phi : POD basis (n x r)

# Returns
(A_hat, H_hat, c_hat) for the ROM ODE:
    a_dot = A_hat*a + H_hat*(a kron a) + c_hat
"""
function galerkin_project(A::Matrix{Float64}, H::Matrix{Float64},
                          c::Vector{Float64}, Phi::Matrix{Float64})
    r = size(Phi, 2)
    A_hat = Phi' * A * Phi
    H_hat = Phi' * H * kron(Phi, Phi)
    c_hat = Phi' * c
    return A_hat, H_hat, c_hat
end

"""
    rom_rhs!(da, a, (A_hat, H_hat, c_hat), t)

In-place RHS for the reduced-order model. Pass to `ODEProblem`.

    a_dot = A_hat*a + H_hat*(a kron a) + c_hat
"""
function rom_rhs!(da, a, params, t)
    A_hat, H_hat, c_hat = params
    da .= A_hat * a .+ H_hat * kron(a, a) .+ c_hat
end