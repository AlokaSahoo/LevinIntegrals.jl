#=
    collocation.jl

    Chebyshev-Lobatto collocation infrastructure for the Levin method.

    Provides:
    - Chebyshev-Lobatto (Clenshaw-Curtis) node generation
    - Affine mapping to arbitrary intervals [a, b]
    - Chebyshev polynomial basis matrix via three-term recurrence
    - Spectral differentiation matrix via the exact barycentric formula
=#

"""
    chebyshev_lobatto_nodes(n::Integer)

Compute `n` Chebyshev-Lobatto (Gauss-Lobatto-Chebyshev) points on ``[-1, 1]``.

The nodes are defined as:
```math
x_j = \\cos\\!\\left(\\frac{\\pi j}{n - 1}\\right), \\quad j = 0, 1, \\ldots, n-1
```
and are returned in **descending** order (from ``+1`` to ``-1``), which is the
standard spectral convention.

# Arguments
- `n::Integer`: number of collocation points (must be ≥ 2).

# Returns
- `Vector{Float64}`: the `n` Chebyshev-Lobatto nodes.

# Examples
```julia
julia> chebyshev_lobatto_nodes(5)
5-element Vector{Float64}:
  1.0
  0.7071067811865476
  0.0
 -0.7071067811865476
 -1.0
```
"""
function chebyshev_lobatto_nodes(n::Integer)
    n < 2 && throw(ArgumentError("Number of nodes must be ≥ 2, got n = $n"))
    return [cos(π * j / (n - 1)) for j in 0:(n - 1)]
end

"""
    map_nodes(nodes, a, b)

Affine map from the reference interval ``[-1, 1]`` to ``[a, b]``.

```math
x \\mapsto \\frac{b - a}{2} x + \\frac{a + b}{2}
```

# Arguments
- `nodes`: vector of points in ``[-1, 1]``.
- `a`, `b`: endpoints of the target interval.

# Returns
- `Vector`: the mapped nodes in ``[a, b]``.
"""
function map_nodes(nodes, a, b)
    return @. (b - a) / 2 * nodes + (a + b) / 2
end

"""
    chebyshev_basis_matrix(nodes, n::Integer)

Evaluate the first `n` Chebyshev polynomials ``T_0, T_1, \\ldots, T_{n-1}``
at each point in `nodes`, returning the basis (Vandermonde-like) matrix.

Uses the three-term recurrence:
```math
T_0(x) = 1,\\quad T_1(x) = x,\\quad T_{k+1}(x) = 2x\\,T_k(x) - T_{k-1}(x).
```

# Arguments
- `nodes`: vector of evaluation points.
- `n::Integer`: number of basis functions.

# Returns
- `Matrix{Float64}` of size `(length(nodes), n)` where entry `(i, j)` is ``T_{j-1}(x_i)``.
"""
function chebyshev_basis_matrix(nodes, n::Integer)
    m = length(nodes)
    B = zeros(m, n)

    # T_0(x) = 1
    B[:, 1] .= 1.0

    if n ≥ 2
        # T_1(x) = x
        B[:, 2] .= nodes
    end

    # Three-term recurrence: T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
    @inbounds for k in 3:n
        @. B[:, k] = 2.0 * nodes * B[:, k - 1] - B[:, k - 2]
    end

    return B
end

"""
    chebyshev_differentiation_matrix(n::Integer)

Compute the `n × n` Chebyshev spectral differentiation matrix on the
Chebyshev-Lobatto grid.

Uses the exact barycentric formula (see Weideman & Reddy, 2000;
Trefethen, *Spectral Methods in MATLAB*, Ch. 6):

```math
D_{ij} = \\frac{c_i}{c_j} \\frac{(-1)^{i+j}}{x_i - x_j}, \\quad i \\neq j
```
```math
D_{ii} = -\\sum_{j \\neq i} D_{ij}
```

where ``c_0 = c_{n-1} = 2`` and ``c_j = 1`` otherwise.

# Arguments
- `n::Integer`: number of collocation points (must be ≥ 2).

# Returns
- `Matrix{Float64}` of size `(n, n)`: the spectral differentiation matrix.
"""
function chebyshev_differentiation_matrix(n::Integer)
    n < 2 && throw(ArgumentError("Need n ≥ 2 for differentiation matrix, got n = $n"))

    x = chebyshev_lobatto_nodes(n)

    # Barycentric weights: c_0 = c_{n-1} = 2, c_j = 1 otherwise
    c = ones(n)
    c[1] = 2.0
    c[n] = 2.0

    D = zeros(n, n)

    @inbounds for i in 1:n
        row_sum = 0.0
        for j in 1:n
            if i != j
                Dij = (c[i] / c[j]) * (-1)^(i + j) / (x[i] - x[j])
                D[i, j] = Dij
                row_sum += Dij
            end
        end
        # Diagonal: D_{ii} = -∑_{j≠i} D_{ij}  (negative sum trick)
        D[i, i] = -row_sum
    end

    return D
end
