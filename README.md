# LevinIntegrals

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://AlokaSahoo.github.io/LevinIntegrals.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AlokaSahoo.github.io/LevinIntegrals.jl/dev/)
[![Build Status](https://github.com/AlokaSahoo/LevinIntegrals.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/AlokaSahoo/LevinIntegrals.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/AlokaSahoo/LevinIntegrals.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AlokaSahoo/LevinIntegrals.jl)

The package provides an implementation of the Levin method for general-purpose integration of Levin-type integrals, with a special focus on applications in Atomic physics.

The Levin-type integrals have the following form:
```math
I = \int_{a}^{b} f(x) e^{i g(x)} \mathrm{d}x
```
where $f(x)$ is a slowly varying amplitude function and $g(x)$ is a rapidly varying phase function.

## High-Level API Calls

The package provides two high-level API calls to evaluate the Levin integral. By default, these functions can take the amplitude function `f` and the phase function `g` directly.

### 1. Adaptive Levin Integration

The adaptive routine automatically subdivides the integration domain to ensure convergence and accuracy, making it robust for complex integrands.

```julia
using LevinIntegrals

# Define the functions
f(x) = 1.0
g(x) = 100.0 * x

# Integration limits
a, b = 0.0, 1.0

# Compute the integral adaptively
result = levin_integrate_adaptive(f, g, a, b)
```

### 2. Levin Integration with (Optional) Fixed Intervals

You can also use the standard integration routine, which can operate over a fixed number of intervals.

```julia
using LevinIntegrals

# Define the functions
f(x) = 1.0
g(x) = 100.0 * x

# Integration limits
a, b = 0.0, 1.0

# Compute the integral with a single fixed interval (n=1 is the default)
result_single = levin_integrate(f, g, a, b; n=1)

# Or divide the domain into multiple fixed sub-intervals (composite rule)
result_multi = levin_integrate(f, g, a, b; n=10)
```

## Explicitly Providing the Phase Derivative `g'`

While providing `f` and `g` is convenient, in physics applications particular in Strong-Field Approximation (SFA) [see the [Applications in SFA](https://AlokaSahoo.github.io/LevinIntegrals.jl/dev/applications_sfa/)] the phase function is an integral and the derivative of the phase function has an algebraic expression. Thus, it is more convenient (also efficient) to provide the derivative of the phase function for the Levin method.

```julia
using LevinIntegrals

# Define the functions
f(x) = 1.0
g(x) = 100.0 * x
g_prime(x) = 100.0  # Exact algebraic expression for the derivative of g

# Integration limits
a, b = 0.0, 1.0

# Fixed intervals with g' (e.g., n=1 sub-interval)
val_fixed = levin_integrate(f, g, g_prime, a, b; n=1)

# Adaptive integration with g'
val_adapt = levin_integrate_adaptive(f, g, g_prime, a, b)
```

For more details on the functions and their keyword arguments, please see the [API Reference](https://AlokaSahoo.github.io/LevinIntegrals.jl/dev/api) and [Basic Usage](https://AlokaSahoo.github.io/LevinIntegrals.jl/dev/basic_usage) pages in the documentation.
