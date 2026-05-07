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
p'(x_j) + \\alpha(x_j)\\, p(x_j) = f(x_j), \\quad j = 0, \\ldots, n-1
```

Expanding ``p(x) = \\sum_k c_k T_k(x)`` and collocating yields the matrix
equation ``A \\mathbf{c} = \\mathbf{f}`` with:

```math
A = D \\cdot B + \\operatorname{diag}(\\boldsymbol{\\alpha}) \\cdot B
```

The system is solved using QR factorization for numerical stability.

# Arguments
- `f_vals::AbstractVector`: values of the amplitude function ``f(x_j)`` at the
  collocation nodes (length `n`).
- `D::AbstractMatrix`: the `n × n` spectral differentiation matrix.
- `B::AbstractMatrix`: the `n × n` Chebyshev basis matrix evaluated at the nodes.
- `alpha_vals::AbstractVector`: values of the oscillator ODE coefficient
  ``\\alpha(x_j)`` at the collocation nodes (e.g., ``i g'(x_j)``).

# Returns
- `Vector`: coefficient vector ``\\mathbf{c}`` of length `n` such that
  ``p(x) = \\sum_k c_k T_k(x)`` satisfies the Levin equation at the nodes.

# Notes
- For exponential oscillators ``w(x) = e^{ig(x)}``, set ``\\alpha(x) = i g'(x)``.
- The solve uses `qr` (column-pivoted QR) via `A \\ f_vals`, which falls back
  to the most stable factorization available in LinearAlgebra.
"""
function levin_collocation_solve(f_vals::AbstractVector,
                                  D::AbstractMatrix,
                                  B::AbstractMatrix,
                                  alpha_vals::AbstractVector)
    n = length(f_vals)

    # Assemble the collocation matrix: A = D*B + diag(α)*B
    A = D * B + Diagonal(alpha_vals) * B

    # Solve via QR factorization
    c = qr(A, ColumnNorm()) \ f_vals

    return c
end
