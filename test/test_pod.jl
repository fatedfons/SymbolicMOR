# test/test_pod.jl
using Test
using LinearAlgebra
using SymbolicMOR

@testset "Phase 2 - POD Basis" begin

    @testset "Basis is orthonormal" begin
        X = randn(20, 200)   # 20-dim state, 200 snapshots
        Phi, sigma, energy = compute_pod_basis(X, 5)
        @test size(Phi) == (20, 5)
        @test norm(Phi' * Phi - I) < 1e-12   # columns are orthonormal
    end

    @testset "Energy threshold selects correct rank" begin
        # Low-rank data: rank 3
        U, _, V = svd(randn(10, 10))
        X = U[:, 1:3] * Diagonal([100.0, 10.0, 1.0]) * V[:, 1:3]'
        Phi, sigma, energy = compute_pod_basis(X; energy_threshold=0.999)
        @test size(Phi, 2) <= 5   # should retain very few modes
        @test energy >= 0.999
    end

    @testset "Galerkin projection reduces dimension" begin
        n, r = 10, 3
        Phi = Matrix(qr(randn(n, r)).Q)[:, 1:r]
        A = randn(n, n)
        H = randn(n, n^2)
        c = randn(n)
        A_hat, H_hat, c_hat = galerkin_project(A, H, c, Phi)
        @test size(A_hat) == (r, r)
        @test size(H_hat) == (r, r^2)
        @test length(c_hat) == r
    end

end