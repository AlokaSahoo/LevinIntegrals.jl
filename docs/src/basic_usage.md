# Basic Usage

The present implementation of Levin method has two different approaches

## Integrating with only `f` and `g`

If you only provide the amplitude function `f` and the phase function `g`, the package will automatically compute or approximate the derivative of `g`.

```julia
using LevinIntegrals

# Amplitude function
f(x) = 1.0

# Rapidly oscillating phase
g(x) = 100.0 * x

# Limits
a, b = 0.0, 1.0

# Calculate the integral
val = levin_integrate(f, g, a, b)
```

## Integrating with explicitly given `g'`

For better numerical stability and performance, it is advisable to supply the derivative of the phase function explicitly:

```julia
using LevinIntegrals

# Functions
f(x) = 1.0
g(x) = 100.0 * x
g_prime(x) = 100.0 # Derivative of g

# Limits
a, b = 0.0, 1.0

# Calculate the integral
val = levin_integrate(f, g, g_prime, a, b)
```

## Optional Keyword Arguments

You can change the polynomial order (number of collocation points per panel) and the number of sub-intervals (panels) by providing the `k` and `n` keyword arguments.

* `k` (default: 16): Number of collocation points per panel.
* `n` (default: 1): Number of equal sub-intervals to divide $[a, b]$ into (composite Levin rule).

```julia
using LevinIntegrals

f(x) = 1.0
g(x) = 100.0 * x
g_prime(x) = 100.0
a, b = 0.0, 1.0

# Using 32 collocation points and dividing the domain into 4 sub-intervals
val = levin_integrate(f, g, g_prime, a, b; k=32, n=4)
```

## Levin Solvers

By default, the collocation system is solved using a column-pivoted QR factorization (`QRSolver()`). However, you can choose different solver strategies using the `solver` keyword argument. Available solvers include `QRSolver`, `LUSolver`, and `TSVDSolver` (Truncated SVD).

You can find more details directly from the Julia REPL using the `?` help mode:

```julia
help?> ODESolver
```

Here is an example demonstrating how to specify the solver:

```julia
using LevinIntegrals

f(x) = 1.0
g(x) = 100.0 * x
g_prime(x) = 100.0
a, b = 0.0, 1.0

# Using the QR solver ( default option. a balanced approach)
val_lu = levin_integrate(f, g, g_prime, a, b; solver=QRSolver())

# Using the LU solver (slightly faster for well-conditioned systems)
val_lu = levin_integrate(f, g, g_prime, a, b; solver=LUSolver())

# Using the Truncated SVD solver (more stable for near-singular systems)
val_tsvd = levin_integrate(f, g, g_prime, a, b; solver=TSVDSolver())
```
