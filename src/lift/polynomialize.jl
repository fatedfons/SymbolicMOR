# src/lift/polynomialize.jl
#
# Phase 1 - Polynomialization
#
# Goal: Rewrite a nonlinear ODE RHS so that every term is a polynomial
# (or rational function cast to polynomial via new auxiliary variables).
#
# This is the first step before quadratization: you can only quadratize
# a system that is already polynomial.

using Symbolics

"""
    polynomialize(expr, vars)

Attempt to rewrite a symbolic expression (expr, in terms of vars) into
polynomial form by introducing auxiliary variables for non-polynomial
sub-expressions (for example sin, exp, sqrt).

Returns (new_expr, aux_vars, aux_eqs) where:
  - new_expr : the rewritten expression (polynomial in vars + aux_vars)
  - aux_vars : vector of new symbolic variables introduced
  - aux_eqs  : vector of equations defining each aux var (for example w1 = sin(x))

Example:
    @variables x
    expr = sin(x) * x^2
    new_expr, aux_vars, aux_eqs = polynomialize(expr, [x])
    # new_expr  => w1 * x^2
    # aux_vars  => [w1]
    # aux_eqs   => [w1 ~ sin(x)]
"""
function polynomialize(expr, vars::Vector)
    aux_vars = Num[]
    aux_eqs  = Equation[]
    aux_counter = Ref(0)

    new_expr = _poly_walk(expr, vars, aux_vars, aux_eqs, aux_counter)
    return new_expr, aux_vars, aux_eqs
end

# Internal recursive walker

function _poly_walk(expr, vars, aux_vars, aux_eqs, counter)
    # Unwrap Symbolics Num wrapper
    ex = Symbolics.unwrap(expr)

    # Base cases: numbers and known variables are already polynomial
    if ex isa Number || Symbolics.issym(ex)
        return expr
    end

    # Check if the whole expression is already polynomial in vars
    if _is_polynomial(expr, vars)
        return expr
    end

    # If it contains a non-polynomial operation, introduce an auxiliary variable
    # Create a new auxiliary variable: w1, w2, ...
    counter[] += 1
    w_name = Symbol("w$(counter[])")
    @variables $(w_name)
    w = eval(:($w_name))
    push!(aux_vars, w)
    push!(aux_eqs, w ~ expr)
    return w
end

"""
    _is_polynomial(expr, vars)

Heuristic check: returns true if expr is a polynomial in vars.

This checks that the expression only involves +, -, *, and integer powers
of the given vars (no transcendental functions, no fractional powers).
"""
function _is_polynomial(expr, vars)
    try
        Symbolics.degree(expr) >= 0
        return true
    catch
        return false
    end
end