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
    chebyshev_lobatto_nodes(k::Integer)

Compute `k` Chebyshev-Lobatto (Gauss-Lobatto-Chebyshev) points on ``[-1, 1]``.

The nodes are defined as:
```math
x_j = \\cos\\!\\left(\\frac{\\pi j}{k - 1}\\right), \\quad j = 0, 1, \\ldots, k-1
```
and are returned in **descending** order (from ``+1`` to ``-1``), which is the
standard spectral convention.

# Arguments
- `k::Integer`: number of collocation points (must be ≥ 2).

# Returns
- `Vector{Float64}`: the `k` Chebyshev-Lobatto nodes.

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
function chebyshev_lobatto_nodes(k::Integer)
    k < 2 && throw(ArgumentError("Number of nodes must be ≥ 2, got k = $k"))
    return [cos(π * j / (k - 1)) for j in 0:(k - 1)]
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
    chebyshev_basis_matrix(nodes, k::Integer)

Evaluate the first `k` Chebyshev polynomials ``T_0, T_1, \\ldots, T_{k-1}``
at each point in `nodes`, returning the basis (Vandermonde-like) matrix.

Uses the three-term recurrence:
```math
T_0(x) = 1,\\quad T_1(x) = x,\\quad T_{j+1}(x) = 2x\\,T_j(x) - T_{j-1}(x).
```

# Arguments
- `nodes`: vector of evaluation points.
- `k::Integer`: number of basis functions.

# Returns
- `Matrix{Float64}` of size `(length(nodes), k)` where entry `(i, j)` is ``T_{j-1}(x_i)``.
"""
function chebyshev_basis_matrix(nodes, k::Integer)
    m = length(nodes)
    B = zeros(m, k)

    # T_0(x) = 1
    B[:, 1] .= 1.0

    if k ≥ 2
        # T_1(x) = x
        B[:, 2] .= nodes
    end

    # Three-term recurrence: T_{j+1}(x) = 2x T_j(x) - T_{j-1}(x)
    @inbounds for j in 3:k
        @. B[:, j] = 2.0 * nodes * B[:, j - 1] - B[:, j - 2]
    end

    return B
end

"""
    chebyshev_differentiation_matrix(k::Integer)

Compute the `k × k` Chebyshev spectral differentiation matrix on the
Chebyshev-Lobatto grid.

Uses the exact barycentric formula (see Weideman & Reddy, 2000;
Trefethen, *Spectral Methods in MATLAB*, Ch. 6):

```math
D_{ij} = \\frac{c_i}{c_j} \\frac{(-1)^{i+j}}{x_i - x_j}, \\quad i \\neq j
```
```math
D_{ii} = -\\sum_{j \\neq i} D_{ij}
```

where ``c_0 = c_{k-1} = 2`` and ``c_j = 1`` otherwise.

# Arguments
- `k::Integer`: number of collocation points (must be ≥ 2).

# Returns
- `Matrix{Float64}` of size `(k, k)`: the spectral differentiation matrix.
"""
function chebyshev_differentiation_matrix(k::Integer)
    k < 2 && throw(ArgumentError("Need k ≥ 2 for differentiation matrix, got k = $k"))

    x = chebyshev_lobatto_nodes(k)

    # Barycentric weights: c_0 = c_{k-1} = 2, c_j = 1 otherwise
    c = ones(k)
    c[1] = 2.0
    c[k] = 2.0

    D = zeros(k, k)

    @inbounds for i in 1:k
        row_sum = 0.0
        for j in 1:k
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
