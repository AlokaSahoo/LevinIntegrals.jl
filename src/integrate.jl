#=
    integrate.jl

    High-level user-facing integration routines for the Levin method.

    Provides:
    - _levin_core        : shared internal helper (nodes → solve → endpoint p values)
    - levin_integrate    : fixed-order Levin integration (three dispatch methods)
    - levin_integrate_adaptive : adaptive h-refinement Levin integration
=#

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _levin_core(f, g_prime, a, b, n) -> (p_a, p_b, phys_nodes)

Internal workhorse shared by all `levin_integrate` dispatch methods.

Performs steps 1–5 of the Levin collocation algorithm:
1. Generate Chebyshev-Lobatto nodes on ``[-1, 1]`` and map to ``[a, b]``.
2. Evaluate ``f(x_j)`` and ``\\alpha(x_j) = i g'(x_j)``.
3. Build basis matrix ``B`` and scaled differentiation matrix ``D``.
4. Solve the collocation system.
5. Evaluate the antiderivative ``p(x) = B \\mathbf{c}`` at the endpoints.

Returns `(p_a, p_b, phys_nodes)` — the caller handles the oscillator evaluation.
"""
function _levin_core(f, g_prime, a, b, n::Integer)
    # 1. Nodes
    ref_nodes  = chebyshev_lobatto_nodes(n)
    phys_nodes = map_nodes(ref_nodes, a, b)

    # 2. Function values
    f_vals     = complex.(f.(phys_nodes))
    alpha_vals = im .* complex.(g_prime.(phys_nodes))

    # 3. Basis and differentiation matrices
    B = chebyshev_basis_matrix(ref_nodes, n)
    D = chebyshev_differentiation_matrix(n)
    D_scaled = D * (2.0 / (b - a))

    # 4. Solve
    c = levin_collocation_solve(f_vals, D_scaled, B, alpha_vals)

    # 5. Endpoint antiderivative values
    #    Index 1 → cos(0) = +1 → mapped to b
    #    Index n → cos(π) = -1 → mapped to a
    p_b = dot(B[1, :], c)
    p_a = dot(B[n, :], c)

    return p_a, p_b, phys_nodes
end

"""
    _numerical_derivative(g, x)

Compute ``g'(x)`` via central finite differences:
```math
g'(x) \\approx \\frac{g(x+h) - g(x-h)}{2h}
```
with step size ``h = \\varepsilon^{1/3} \\max(1, |x|)`` for optimal balance of
truncation and roundoff error (yields ``\\sim \\varepsilon^{2/3} \\approx 4 \\times 10^{-11}``
accuracy). Internal helper — not exported.
"""
function _numerical_derivative(g, x)
    h = cbrt(eps(Float64)) * max(1.0, abs(x))
    return (g(x + h) - g(x - h)) / (2h)
end

"""
    _integrate_gprime(g_prime, a, x; nquad=64)

Numerically compute ``g(x) = \\int_a^x g'(t)\\, dt`` using a composite
trapezoidal rule with `nquad` sub-intervals. Exact for linear `g'`.
Internal helper — not exported.
"""
function _integrate_gprime(g_prime, a, x; nquad::Integer = 64)
    x ≈ a && return 0.0
    t = range(a, x, length = nquad + 1)
    vals = g_prime.(t)
    h = (x - a) / nquad
    return h * (vals[1] / 2 + @views(sum(vals[2:end-1])) + vals[end] / 2)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-order Levin integration — three dispatch methods
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levin_integrate(f, g, g_prime, a, b; n=16)

Compute the oscillatory integral

```math
I = \\int_a^b f(x)\\, e^{i\\,g(x)}\\, dx
```

using the Levin collocation method with `n` Chebyshev-Lobatto points.
Both the phase function `g(x)` and its derivative `g'(x)` are provided
explicitly for maximum accuracy.

# Arguments
- `f`: amplitude function ``f(x)``.
- `g`: phase function ``g(x)`` — used to evaluate ``w(x) = e^{ig(x)}`` at endpoints.
- `g_prime`: derivative of the phase, ``g'(x)`` — used in the collocation system.
- `a`, `b`: integration limits (`a < b`).

# Keyword Arguments
- `n::Integer = 16`: number of collocation points.

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
ω = 100.0
result = levin_integrate(x -> 1.0, x -> ω*x, x -> ω, 0.0, 1.0; n=32)
exact  = (exp(im*ω) - 1) / (im*ω)
abs(result - exact)  # ≈ 0 to machine precision
```
"""
function levin_integrate(f, g, g_prime, a, b; n::Integer = 16)
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    n < 2 && throw(ArgumentError("Need at least 2 collocation points, got n = $n"))

    p_a, p_b, _ = _levin_core(f, g_prime, a, b, n)

    w_a = exp(im * g(a))
    w_b = exp(im * g(b))

    return p_b * w_b - p_a * w_a
end

"""
    levin_integrate(f, g, a, b; n=16)

Compute the oscillatory integral when only the phase function `g(x)` is
available. The derivative `g'(x)` is computed internally via central finite
differences.

# Arguments
- `f`: amplitude function ``f(x)``.
- `g`: phase function ``g(x)``.
- `a`, `b`: integration limits (`a < b`).

# Keyword Arguments
- `n::Integer = 16`: number of collocation points.

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
ω = 100.0
result = levin_integrate(x -> 1.0, x -> ω*x, 0.0, 1.0; n=32)
```
"""
function levin_integrate(f, g, a::Real, b::Real; n::Integer = 16)
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    n < 2 && throw(ArgumentError("Need at least 2 collocation points, got n = $n"))

    # Compute g' numerically from g
    g_prime = x -> _numerical_derivative(g, x)

    p_a, p_b, _ = _levin_core(f, g_prime, a, b, n)

    # Exact oscillator evaluation at endpoints (g is known)
    w_a = exp(im * g(a))
    w_b = exp(im * g(b))

    return p_b * w_b - p_a * w_a
end

# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Levin integration — h-refinement via recursive bisection
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levin_integrate_adaptive(f, g, g_prime, a, b; n=16, atol=1e-12, rtol=1e-12, maxdepth=20)

Adaptively compute the oscillatory integral

```math
I = \\int_a^b f(x)\\, e^{i\\,g(x)}\\, dx
```

using recursive bisection (h-refinement) of the Levin collocation method.

On each sub-interval, the method compares a coarse estimate (single panel) with
a fine estimate (two half-panels). If they disagree by more than the requested
tolerance, the sub-interval is bisected and each half is refined recursively.

# Arguments
- `f`: amplitude function ``f(x)``.
- `g`: phase function ``g(x)`` — used for exact oscillator evaluation.
- `g_prime`: derivative of the phase, ``g'(x)``.
- `a`, `b`: integration limits (`a < b`).

# Keyword Arguments
- `n::Integer = 16`: collocation points per panel.
- `atol::Real = 1e-12`: absolute error tolerance.
- `rtol::Real = 1e-12`: relative error tolerance.
- `maxdepth::Integer = 20`: maximum recursion depth.

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
# Adaptive handles a rapidly-varying amplitude well
result = levin_integrate_adaptive(
    x -> exp(5x), x -> 100.0*x, x -> 100.0, 0.0, 2.0)
```
"""
function levin_integrate_adaptive(f, g, g_prime, a, b;
                                   n::Integer = 16,
                                   atol::Real = 1e-12,
                                   rtol::Real = 1e-12,
                                   maxdepth::Integer = 20)
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    return _adaptive_levin(f, g, g_prime, a, b, n, atol, rtol, maxdepth, 0)
end

"""
    levin_integrate_adaptive(f, g, a::Real, b::Real; n=16, atol=1e-12, rtol=1e-12, maxdepth=20)

Adaptive Levin integration when only the phase function `g(x)` is available.
The derivative `g'(x)` is computed internally via central finite differences.

See the 5-argument method for full documentation.
"""
function levin_integrate_adaptive(f, g, a::Real, b::Real;
                                   n::Integer = 16,
                                   atol::Real = 1e-12,
                                   rtol::Real = 1e-12,
                                   maxdepth::Integer = 20)
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))

    # Compute g' numerically from g
    g_prime = x -> _numerical_derivative(g, x)
    return _adaptive_levin(f, g, g_prime, a, b, n, atol, rtol, maxdepth, 0)
end

"""
    _adaptive_levin(f, g, g_prime, a, b, n, atol, rtol, maxdepth, depth)

Recursive bisection engine for adaptive Levin integration.
"""
function _adaptive_levin(f, g, g_prime, a, b, n, atol, rtol, maxdepth, depth)
    # Coarse estimate: single panel [a, b]
    I_coarse = levin_integrate(f, g, g_prime, a, b; n=n)

    # Fine estimate: two half-panels
    m = (a + b) / 2
    I_left  = levin_integrate(f, g, g_prime, a, m; n=n)
    I_right = levin_integrate(f, g, g_prime, m, b; n=n)
    I_fine  = I_left + I_right

    # Error estimate
    err = abs(I_fine - I_coarse)
    tol = max(atol, rtol * abs(I_fine))

    if err ≤ tol || depth ≥ maxdepth
        return I_fine
    else
        # Recurse on each half
        I_left_refined  = _adaptive_levin(f, g, g_prime, a, m, n,
                                           atol / 2, rtol, maxdepth, depth + 1)
        I_right_refined = _adaptive_levin(f, g, g_prime, m, b, n,
                                           atol / 2, rtol, maxdepth, depth + 1)
        return I_left_refined + I_right_refined
    end
end
