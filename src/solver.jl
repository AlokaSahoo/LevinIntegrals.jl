#=
    solver.jl

    Levin collocation system assembly and solution.

    Given the Chebyshev-Lobatto differentiation matrix D, the basis matrix B,
    and the oscillator coefficient α(x) = ig'(x), this module constructs the
    collocation system:

        A c = f

    where A = D·B + diag(α) · B, and solves it via QR factorization.
=#

"""
    levin_collocation_solve(f_vals, D, B, alpha_vals)

Assemble and solve the Levin collocation linear system.

The Levin differential equation at the collocation nodes is:

```math
p'(x_j) + \\alpha(x_j)\\, p(x_j) = f(x_j), \\quad j = 0, \\ldots, k-1
```

Expanding ``p(x) = \\sum_j c_j T_j(x)`` and collocating yields the matrix
equation ``A \\mathbf{c} = \\mathbf{f}`` with:

```math
A = D \\cdot B + \\operatorname{diag}(\\boldsymbol{\\alpha}) \\cdot B
```

The system is solved using QR factorization for numerical stability.

# Arguments
- `f_vals::AbstractVector`: values of the amplitude function ``f(x_j)`` at the
  collocation nodes (length `k`).
- `D::AbstractMatrix`: the `k × k` spectral differentiation matrix.
- `B::AbstractMatrix`: the `k × k` Chebyshev basis matrix evaluated at the nodes.
- `alpha_vals::AbstractVector`: values of the oscillator ODE coefficient
  ``\\alpha(x_j)`` at the collocation nodes (e.g., ``i g'(x_j)``).

# Returns
- `Vector`: coefficient vector ``\\mathbf{c}`` of length `k` such that
  ``p(x) = \\sum_j c_j T_j(x)`` satisfies the Levin equation at the nodes.

# Notes
- For exponential oscillators ``w(x) = e^{ig(x)}``, set ``\\alpha(x) = i g'(x)``.
- The solve uses `qr` (column-pivoted QR) via `A \\ f_vals`, which falls back
  to the most stable factorization available in LinearAlgebra.
"""
function levin_collocation_solve(f_vals::AbstractVector,
                                  D::AbstractMatrix,
                                  B::AbstractMatrix,
                                  alpha_vals::AbstractVector)
    k = length(f_vals)

    # Assemble the collocation matrix: A = (D + diag(α)) * B  (single multiply)
    A = (D + Diagonal(alpha_vals)) * B

    # Solve via QR factorization
    c = qr(A, ColumnNorm()) \ f_vals

    return c
end
