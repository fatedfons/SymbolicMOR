# test/test_quadratize.jl
#
# Unit tests for Phase 1: Symbolic Quadratization

using Test
using Symbolics
using SymbolicMOR

@testset "Phase 1 - Quadratization" begin

    @testset "Degree-2 expression is unchanged" begin
        @variables x y
        expr = x^2 + x*y + y
        new_expr, new_vars, aux_eqs = quadratize(expr, [x, y])
        @test isempty(aux_eqs)           # no aux vars needed
        @test isequal(new_expr, expr)    # expression unchanged
    end

    @testset "Cubic monomial x^3 is lifted" begin
        @variables x
        expr = x^3
        new_expr, new_vars, aux_eqs = quadratize(expr, [x])
        @test length(aux_eqs) == 1                   # one aux var introduced
        @test length(new_vars) == 2                  # [x, w1]
        # The new expression should be quadratic in new_vars
        max_deg = maximum(Symbolics.degree(new_expr, v) for v in new_vars)
        @test max_deg <= 2
    end

    @testset "Lorenz system lift" begin
        @variables x y z
        sigma, rho, beta = 10.0, 28.0, 8/3

        rhs = [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]

        ls = lift_system([x, y, z], rhs)

        # The Lorenz system is already quadratic, so no aux vars should be needed
        @test length(ls.aux_eqs) == 0
        @test length(ls.lifted_vars) == 3

        # Every RHS term must be degree <= 2
        for (v, f) in zip(ls.lifted_vars, ls.F)
            max_deg = maximum(Symbolics.degree(f, w) for w in ls.lifted_vars; init=0)
            @test max_deg <= 2
        end
    end

    @testset "Quartic system needs lifting" begin
        @variables x y
        # x_dot = x^2*y^2 is degree 4 - must be lifted to quadratic
        rhs = [x^2 * y^2, x - y]
        ls = lift_system([x, y], rhs)

        for f in ls.F
            max_deg = maximum(Symbolics.degree(f, w) for w in ls.lifted_vars; init=0)
            @test max_deg <= 2
        end
        @test length(ls.aux_eqs) >= 1   # at least one aux var introduced
    end

end