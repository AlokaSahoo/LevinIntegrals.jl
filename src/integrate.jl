#=
    integrate.jl

    High-level user-facing integration routines for the Levin method.

    Provides:
    - _levin_core        : shared internal helper (nodes → solve → endpoint p values)
    - levin_integrate    : fixed-order Levin integration (three dispatch methods)
    - levin_integrate_adaptive : adaptive h-refinement Levin integration

    All public functions accept an optional `solver::LevinSolver` keyword
    (default `QRSolver()`) that selects the linear-system factorization strategy.
=#

# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _levin_core(f, g_prime, a, b, k, solver) -> (p_a, p_b, phys_nodes)

Internal workhorse shared by all `levin_integrate` dispatch methods.

Performs steps 1–5 of the Levin collocation algorithm:
1. Generate Chebyshev-Lobatto nodes on ``[-1, 1]`` and map to ``[a, b]``.
2. Evaluate ``f(x_j)`` and ``\\alpha(x_j) = i g'(x_j)``.
3. Build basis matrix ``B`` and scaled differentiation matrix ``D``.
4. Solve the collocation system using `solver`.
5. Evaluate the antiderivative ``p(x) = B \\mathbf{c}`` at the endpoints.

Returns `(p_a, p_b, phys_nodes)` — the caller handles the oscillator evaluation.
"""
function _levin_core(f::F, g_prime::GP, a::Real, b::Real, k::Integer, solver::LevinSolver) where {F, GP}
    # 1. Nodes
    ref_nodes  = chebyshev_lobatto_nodes(k)
    phys_nodes = map_nodes(ref_nodes, a, b)

    # 2. Function values
    f_vals     = complex.(f.(phys_nodes))
    alpha_vals = im .* complex.(g_prime.(phys_nodes))

    # 3. Basis and differentiation matrices
    B = chebyshev_basis_matrix(ref_nodes, k)
    D = chebyshev_differentiation_matrix(k)
    D_scaled = D * (2.0 / (b - a))

    # 4. Solve
    c = levin_collocation_solve(f_vals, D_scaled, B, alpha_vals, solver)

    # 5. Endpoint antiderivative values
    #    Index 1 → cos(0) = +1 → mapped to b
    #    Index k → cos(π) = -1 → mapped to a
    p_b = dot(B[1, :], c)
    p_a = dot(B[k, :], c)

    return p_a, p_b, phys_nodes
end

"""
    _levin_core_from_g(f, g, a, b, k, solver) -> (p_a, p_b, g_a, g_b)

Internal workhorse for the g-only Levin integration dispatch.

Identical to [`_levin_core`](@ref) except that the derivative ``g'(x_j)``
is computed **spectrally** via the Chebyshev differentiation matrix
rather than requiring an explicit callable.  This gives spectral (exponential)
convergence of the derivative instead of the ``\\varepsilon^{2/3}`` ceiling of
finite differences.

Returns `(p_a, p_b, g_a, g_b)` — the endpoint antiderivative values and the
phase values ``g(a), g(b)`` (already available from evaluating `g` at the nodes).
"""
function _levin_core_from_g(f::F, g::G, a::Real, b::Real, k::Integer, solver::LevinSolver) where {F, G}
    # 1. Nodes
    ref_nodes  = chebyshev_lobatto_nodes(k)
    phys_nodes = map_nodes(ref_nodes, a, b)

    # 2. Function values
    f_vals = complex.(f.(phys_nodes))
    g_vals = g.(phys_nodes)

    # 3. Basis and differentiation matrices
    B = chebyshev_basis_matrix(ref_nodes, k)
    D = chebyshev_differentiation_matrix(k)
    D_scaled = D * (2.0 / (b - a))

    # 4. Spectral derivative of g — no finite differences needed
    g_prime_vals = D_scaled * g_vals
    alpha_vals   = im .* complex.(g_prime_vals)

    # 5. Solve
    c = levin_collocation_solve(f_vals, D_scaled, B, alpha_vals, solver)

    # 6. Endpoint antiderivative values
    #    Index 1 → cos(0) = +1 → mapped to b
    #    Index k → cos(π) = -1 → mapped to a
    p_b = dot(B[1, :], c)
    p_a = dot(B[k, :], c)

    # g at endpoints (already computed)
    g_b = g_vals[1]
    g_a = g_vals[k]

    return p_a, p_b, g_a, g_b
end

"""
    _levin_integrate_interval(f, g, a, b, k, solver)

Compute the Levin integral on a single panel ``[a, b]`` with `k` collocation
points, using spectral differentiation for ``g'``. Internal helper — not exported.
"""
function _levin_integrate_interval(f::F, g::G, a::Real, b::Real, k::Integer, solver::LevinSolver) where {F, G}
    p_a, p_b, g_a, g_b = _levin_core_from_g(f, g, a, b, k, solver)

    w_a = exp(im * g_a)
    w_b = exp(im * g_b)

    return p_b * w_b - p_a * w_a
end

"""
    _levin_integrate_interval(f, g, g_prime, a, b, k, solver)

Compute the Levin integral on a single panel ``[a, b]`` with `k` collocation
points. Internal helper — not exported.
"""
function _levin_integrate_interval(f::F, g::G, g_prime::GP, a::Real, b::Real, k::Integer, solver::LevinSolver) where {F, G, GP}
    p_a, p_b, _ = _levin_core(f, g_prime, a, b, k, solver)

    w_a = exp(im * g(a))
    w_b = exp(im * g(b))

    return p_b * w_b - p_a * w_a
end

# ═══════════════════════════════════════════════════════════════════════════════
# Fixed-order Levin integration — three dispatch methods
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levin_integrate(f, g, g_prime, a, b; k=16, n=1, solver=QRSolver())

Compute the oscillatory integral

```math
I = \\int_a^b f(x)\\, e^{i\\,g(x)}\\, dx
```

using the Levin collocation method with `k` Chebyshev-Lobatto points per panel.
Both the phase function `g(x)` and its derivative `g'(x)` are provided
explicitly for maximum accuracy.

When `n > 1`, the domain ``[a, b]`` is divided into `n` equal sub-intervals
and the Levin method is applied independently on each panel with `k`
collocation points, summing the results (composite Levin rule).

# Arguments
- `f`: amplitude function ``f(x)``.
- `g`: phase function ``g(x)`` — used to evaluate ``w(x) = e^{ig(x)}`` at endpoints.
- `g_prime`: derivative of the phase, ``g'(x)`` — used in the collocation system.
- `a`, `b`: integration limits (`a < b`).

# Keyword Arguments
- `k::Integer = 16`: number of collocation points per panel.
- `n::Integer = 1`: number of sub-intervals (panels) to divide ``[a, b]`` into.
- `solver::LevinSolver = QRSolver()`: factorization strategy for the collocation
  system. See [`LevinSolver`](@ref) for available options.

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
ω = 100.0
result = levin_integrate(x -> 1.0, x -> ω*x, x -> ω, 0.0, 1.0; k=32)
exact  = (exp(im*ω) - 1) / (im*ω)
abs(result - exact)  # ≈ 0 to machine precision

# Composite: 4 panels with 8 collocation points each
result = levin_integrate(x -> exp(5x), x -> ω*x, x -> ω, 0.0, 2.0; k=8, n=4)

# LU solver
result = levin_integrate(x -> 1.0, x -> ω*x, x -> ω, 0.0, 1.0; solver=LUSolver())

# Truncated SVD
result = levin_integrate(x -> 1.0, x -> ω*x, x -> ω, 0.0, 1.0; solver=TSVDSolver())
```
"""
function levin_integrate(f::F, g::G, g_prime::GP, a::Real, b::Real;
                          k::Integer = 16,
                          n::Integer = 1,
                          solver::LevinSolver = QRSolver()) where {F, G, GP}
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    k < 2 && throw(ArgumentError("Need at least 2 collocation points, got k = $k"))
    n < 1 && throw(ArgumentError("Need at least 1 sub-interval, got n = $n"))

    if n == 1
        return _levin_integrate_interval(f, g, g_prime, a, b, k, solver)
    end

    # Composite rule: divide [a, b] into n equal panels
    edges = range(a, b, length = n + 1)
    result = zero(ComplexF64)
    for i in 1:n
        result += _levin_integrate_interval(f, g, g_prime, edges[i], edges[i + 1], k, solver)
    end
    return result
end

"""
    levin_integrate(f, g, a, b; k=16, n=1, solver=QRSolver())

Compute the oscillatory integral when only the phase function `g(x)` is
available. The derivative `g'(x)` is computed **spectrally** using the
Chebyshev differentiation matrix — no finite differences are needed.

# Arguments
- `f`: amplitude function ``f(x)``.
- `g`: phase function ``g(x)``.
- `a`, `b`: integration limits (`a < b`).

# Keyword Arguments
- `k::Integer = 16`: number of collocation points per panel.
- `n::Integer = 1`: number of sub-intervals (panels) to divide ``[a, b]`` into.
- `solver::LevinSolver = QRSolver()`: factorization strategy. See [`LevinSolver`](@ref).

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
ω = 100.0
result = levin_integrate(x -> 1.0, x -> ω*x, 0.0, 1.0; k=32)

# With TSVD solver
result = levin_integrate(x -> 1.0, x -> ω*x, 0.0, 1.0; solver=TSVDSolver())
```
"""
function levin_integrate(f::F, g::G, a::Real, b::Real;
                          k::Integer = 16,
                          n::Integer = 1,
                          solver::LevinSolver = QRSolver()) where {F, G}
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    k < 2 && throw(ArgumentError("Need at least 2 collocation points, got k = $k"))
    n < 1 && throw(ArgumentError("Need at least 1 sub-interval, got n = $n"))

    if n == 1
        return _levin_integrate_interval(f, g, a, b, k, solver)
    end

    # Composite rule: divide [a, b] into n equal panels
    edges = range(a, b, length = n + 1)
    result = zero(ComplexF64)
    for i in 1:n
        result += _levin_integrate_interval(f, g, edges[i], edges[i + 1], k, solver)
    end
    return result
end

# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Levin integration — h-refinement via recursive bisection
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levin_integrate_adaptive(f, g, g_prime, a, b; k=16, atol=1e-12, rtol=1e-12, maxdepth=20, solver=QRSolver())

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
- `k::Integer = 16`: collocation points per panel.
- `atol::Real = 1e-12`: absolute error tolerance.
- `rtol::Real = 1e-12`: relative error tolerance.
- `maxdepth::Integer = 20`: maximum recursion depth.
- `solver::LevinSolver = QRSolver()`: factorization strategy. See [`LevinSolver`](@ref).

# Returns
- `ComplexF64`: the value of the integral.

# Examples
```julia
# Adaptive handles a rapidly-varying amplitude well
result = levin_integrate_adaptive(
    x -> exp(5x), x -> 100.0*x, x -> 100.0, 0.0, 2.0)

# With TSVD for maximum stability
result = levin_integrate_adaptive(
    x -> exp(5x), x -> 100.0*x, x -> 100.0, 0.0, 2.0; solver=TSVDSolver())
```
"""
function levin_integrate_adaptive(f::F, g::G, g_prime::GP, a::Real, b::Real;
                                   k::Integer = 16,
                                   atol::Real = 1e-12,
                                   rtol::Real = 1e-12,
                                   maxdepth::Integer = 20,
                                   solver::LevinSolver = QRSolver()) where {F, G, GP}
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    return _adaptive_levin(f, g, g_prime, a, b, k, atol, rtol, maxdepth, 0, solver)
end

"""
    levin_integrate_adaptive(f, g, a::Real, b::Real; k=16, atol=1e-12, rtol=1e-12, maxdepth=20, solver=QRSolver())

Adaptive Levin integration when only the phase function `g(x)` is available.
The derivative `g'(x)` is computed **spectrally** using the Chebyshev
differentiation matrix.

See the 5-argument method for full documentation.
"""
function levin_integrate_adaptive(f::F, g::G, a::Real, b::Real;
                                   k::Integer = 16,
                                   atol::Real = 1e-12,
                                   rtol::Real = 1e-12,
                                   maxdepth::Integer = 20,
                                   solver::LevinSolver = QRSolver()) where {F, G}
    a ≥ b && throw(ArgumentError("Require a < b, got a = $a, b = $b"))
    return _adaptive_levin(f, g, a, b, k, atol, rtol, maxdepth, 0, solver)
end

"""
    _adaptive_levin(f, g, g_prime, a, b, k, atol, rtol, maxdepth, depth, solver)

Recursive bisection engine for adaptive Levin integration.
"""
function _adaptive_levin(f::F, g::G, g_prime::GP, a::Real, b::Real, k::Integer,
                          atol::Real, rtol::Real, maxdepth::Integer, depth::Integer,
                          solver::LevinSolver) where {F, G, GP}
    # Coarse estimate: single panel [a, b]
    I_coarse = _levin_integrate_interval(f, g, g_prime, a, b, k, solver)

    # Fine estimate: two half-panels
    m = (a + b) / 2
    I_left  = _levin_integrate_interval(f, g, g_prime, a, m, k, solver)
    I_right = _levin_integrate_interval(f, g, g_prime, m, b, k, solver)
    I_fine  = I_left + I_right

    # Error estimate
    err = abs(I_fine - I_coarse)
    tol = max(atol, rtol * abs(I_fine))

    if err ≤ tol || depth ≥ maxdepth
        return I_fine
    else
        # Recurse on each half
        I_left_refined  = _adaptive_levin(f, g, g_prime, a, m, k,
                                           atol / 2, rtol, maxdepth, depth + 1, solver)
        I_right_refined = _adaptive_levin(f, g, g_prime, m, b, k,
                                           atol / 2, rtol, maxdepth, depth + 1, solver)
        return I_left_refined + I_right_refined
    end
end

"""
    _adaptive_levin(f, g, a, b, k, atol, rtol, maxdepth, depth, solver)

Recursive bisection engine for adaptive Levin integration using spectral
differentiation for ``g'``.
"""
function _adaptive_levin(f::F, g::G, a::Real, b::Real, k::Integer,
                          atol::Real, rtol::Real, maxdepth::Integer, depth::Integer,
                          solver::LevinSolver) where {F, G}
    # Coarse estimate: single panel [a, b]
    I_coarse = _levin_integrate_interval(f, g, a, b, k, solver)

    # Fine estimate: two half-panels
    m = (a + b) / 2
    I_left  = _levin_integrate_interval(f, g, a, m, k, solver)
    I_right = _levin_integrate_interval(f, g, m, b, k, solver)
    I_fine  = I_left + I_right

    # Error estimate
    err = abs(I_fine - I_coarse)
    tol = max(atol, rtol * abs(I_fine))

    if err ≤ tol || depth ≥ maxdepth
        return I_fine
    else
        # Recurse on each half
        I_left_refined  = _adaptive_levin(f, g, a, m, k,
                                             atol / 2, rtol, maxdepth, depth + 1, solver)
        I_right_refined = _adaptive_levin(f, g, m, b, k,
                                             atol / 2, rtol, maxdepth, depth + 1, solver)
        return I_left_refined + I_right_refined
    end
end
