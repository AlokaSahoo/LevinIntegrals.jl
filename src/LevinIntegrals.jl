module LevinIntegrals

using LinearAlgebra

include("collocation.jl")
include("solver.jl")
include("integrate.jl")

export chebyshev_lobatto_nodes
export chebyshev_differentiation_matrix
export levin_collocation_solve
export levin_integrate
export levin_integrate_adaptive

# Solver strategy types
export LevinSolver
export QRSolver
export LUSolver
export TSVDSolver

end
