#=
    solver.jl

    Levin collocation system assembly and solution.

    Given the Chebyshev-Lobatto differentiation matrix D, the basis matrix B,
    and the oscillator coefficient α(x) = ig'(x), this module constructs the
    collocation system:

        A c = f

    where A = D·B + diag(α) · B, and solves it using the strategy specified by
    a `LevinSolver` instance (QR, LU, or truncated SVD).
=#

# ═══════════════════════════════════════════════════════════════════════════════
# Solver strategy types
# ═══════════════════════════════════════════════════════════════════════════════

"""
    LevinSolver

Abstract supertype for Levin collocation solver strategies.

Choose a concrete subtype and pass it as the `solver` keyword argument to
`levin_integrate` or `levin_integrate_adaptive`:

| Type | Factorization | Characteristics |
|------|--------------|-----------------|
| [`QRSolver`](@ref) | Column-pivoted QR | Default. Robust, fast. |
| [`LUSolver`](@ref) | Partial-pivoting LU | Slightly faster, less regularization. |
| [`TSVDSolver`](@ref) | Truncated SVD | Slowest but maximally stable for near-singular systems. |

# Examples
```julia
using LevinIntegrals

# Default (QR) — no solver kwarg needed
levin_integrate(f, g, g_prime, 0.0, 1.0)

# Explicit QR
levin_integrate(f, g, g_prime, 0.0, 1.0; solver = QRSolver())

# LU
levin_integrate(f, g, g_prime, 0.0, 1.0; solver = LUSolver())

# Truncated SVD with default threshold 1e-14
levin_integrate_adaptive(f, g, 0.0, 1.0; solver = TSVDSolver())

# Truncated SVD with custom threshold
levin_integrate_adaptive(f, g, 0.0, 1.0; solver = TSVDSolver(1e-12))
```
"""
abstract type LevinSolver end

"""
    QRSolver()

Solve the Levin collocation system via **column-pivoted QR** factorization.

This is the default solver. It is backward-compatible with the original
implementation and provides a good balance between speed and numerical stability.
Column pivoting makes it robust against mildly rank-deficient systems.
"""
struct QRSolver <: LevinSolver end

"""
    LUSolver()

Solve the Levin collocation system via **LU** factorization (partial pivoting).

Slightly faster than `QRSolver` for well-conditioned systems, but provides less
regularization. Suitable when the collocation matrix is known to be well-conditioned
(e.g., moderate `k`, smooth amplitude, and non-degenerate phase).
"""
struct LUSolver <: LevinSolver end

"""
    TSVDSolver(tol::Float64 = 1e-14)

Solve the Levin collocation system via **truncated SVD**.

Singular values smaller than `tol * σ_max` are treated as zero, effectively
regularizing near-rank-deficient systems. This is the most numerically stable
strategy but is also the slowest (O(k³) full SVD).

Use this when:
- The collocation matrix is nearly singular (e.g., the phase `g'` is nearly zero
  on the interval, making the Levin ODE ill-conditioned).
- You require the most robust solution at the cost of speed.

# Fields
- `tol::Float64`: relative singular-value truncation threshold (default `1e-14`).

# Examples
```julia
# Default threshold
solver = TSVDSolver()

# Stricter threshold
solver = TSVDSolver(1e-12)
```
"""
struct TSVDSolver <: LevinSolver
    tol::Float64
end
TSVDSolver() = TSVDSolver(1e-14)

# ═══════════════════════════════════════════════════════════════════════════════
# System assembly (shared)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    _assemble_levin_matrix(f_vals, D, B, alpha_vals) -> (A, f_vals)

Assemble the Levin collocation matrix ``A = (D + \\operatorname{diag}(α)) \\cdot B``.
Returns `(A, f_vals)` ready for solving.
"""
@inline function _assemble_levin_matrix(f_vals, D, B, alpha_vals)
    return (D + Diagonal(alpha_vals)) * B
end

# ═══════════════════════════════════════════════════════════════════════════════
# levin_collocation_solve — dispatched on solver type
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levin_collocation_solve(f_vals, D, B, alpha_vals[, solver]) -> Vector

Assemble and solve the Levin collocation linear system ``A \\mathbf{c} = \\mathbf{f}``,
where ``A = (D + \\operatorname{diag}(\\boldsymbol{α})) \\cdot B``.

The `solver` argument selects the factorization strategy (see [`LevinSolver`](@ref)).
Omitting `solver` defaults to [`QRSolver`](@ref).

# Arguments
- `f_vals::AbstractVector`: values of the amplitude function ``f(x_j)`` at the
  collocation nodes (length `k`).
- `D::AbstractMatrix`: the `k × k` scaled spectral differentiation matrix.
- `B::AbstractMatrix`: the `k × k` Chebyshev basis matrix evaluated at the nodes.
- `alpha_vals::AbstractVector`: values of ``α(x_j) = i g'(x_j)`` at the nodes.
- `solver::LevinSolver`: factorization strategy (default `QRSolver()`).

# Returns
- `Vector`: coefficient vector ``\\mathbf{c}`` of length `k`.
"""
function levin_collocation_solve(f_vals::AbstractVector,
                                  D::AbstractMatrix,
                                  B::AbstractMatrix,
                                  alpha_vals::AbstractVector,
                                  ::QRSolver)
    A = _assemble_levin_matrix(f_vals, D, B, alpha_vals)
    return qr(A, ColumnNorm()) \ f_vals
end

function levin_collocation_solve(f_vals::AbstractVector,
                                  D::AbstractMatrix,
                                  B::AbstractMatrix,
                                  alpha_vals::AbstractVector,
                                  ::LUSolver)
    A = _assemble_levin_matrix(f_vals, D, B, alpha_vals)
    return lu(A) \ f_vals
end

function levin_collocation_solve(f_vals::AbstractVector,
                                  D::AbstractMatrix,
                                  B::AbstractMatrix,
                                  alpha_vals::AbstractVector,
                                  s::TSVDSolver)
    A  = _assemble_levin_matrix(f_vals, D, B, alpha_vals)
    F  = svd(A)
    # Truncate singular values below tol * σ_max
    σ_max = F.S[1]   # singular values are sorted descending
    threshold = s.tol * σ_max
    S_inv = map(σ -> σ > threshold ? inv(σ) : zero(σ), F.S)
    # c = V * diag(S_inv) * U' * f_vals
    return F.V * (S_inv .* (F.U' * f_vals))
end

# ── Backward-compatible 4-argument form (defaults to QRSolver) ────────────────
"""
    levin_collocation_solve(f_vals, D, B, alpha_vals)

Backward-compatible form — equivalent to passing `QRSolver()`.
"""
levin_collocation_solve(f_vals::AbstractVector,
                         D::AbstractMatrix,
                         B::AbstractMatrix,
                         alpha_vals::AbstractVector) =
    levin_collocation_solve(f_vals, D, B, alpha_vals, QRSolver())
