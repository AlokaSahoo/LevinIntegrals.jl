# Theory

The Levin method is a powerful technique for evaluating highly oscillatory integrals. Instead of trying to resolve the rapid oscillations directly using standard quadrature rules (which requires an impractically large number of integration points), the Levin method transforms the integration problem into an ordinary differential equation (ODE) problem.

## The Basic Form

Consider a highly oscillatory integral of the form:

```math
I = \int_{a}^{b} f(x) e^{i g(x)} \mathrm{d}x
```

where, $f(x)$ is a slowly varying amplitude function, $g(x)$ is a rapidly varying phase function and $e^{i g(x)}$ represents the highly oscillatory kernel.

## Conversion to an ODE

The core idea of the Levin method is to find an antiderivative $F(x)$ of the integrand that shares the same oscillatory behavior. We assume the antiderivative takes the form:

```math
F(x) = p(x) e^{i g(x)}
```

where $p(x)$ is a non-oscillatory, slowly varying function that we need to determine. 

Taking the derivative of $F(x)$ with respect to $x$ using the product rule yields:

```math
F'(x) = \left[ p'(x) + i g'(x) p(x) \right] e^{i g(x)}
```

By definition, $F'(x)$ must equal the integrand $f(x) e^{i g(x)}$. Equating the terms inside the brackets gives us a linear ordinary differential equation for the unknown function $p(x)$:

```math
p'(x) + i g'(x) p(x) = f(x)
```

If we can solve this ODE for $p(x)$, the value of the definite integral simply becomes the difference of the antiderivative at the boundaries:

```math
I = F(b) - F(a) = p(b) e^{i g(b)} - p(a) e^{i g(a)}
```

## The Collocation Method

To solve the ODE numerically, `LevinIntegrals.jl` employs a spectral collocation method. 

We approximate the slowly varying function $p(x)$ as a linear combination of basis functions. In this package, we use Chebyshev polynomials $B_j(x)$ up to order $k$:

```math
p(x) \approx \sum_{j=1}^{k} c_j B_j(x)
```

We then require the ODE to be satisfied exactly at a set of discrete collocation points $\{x_m\}_{m=1}^k$. For optimal spectral convergence, we use Chebyshev-Lobatto nodes mapped to the integration interval $[a, b]$.

Substituting the basis expansion into the ODE evaluated at the collocation points gives a system of linear equations for the unknown coefficients $c_j$:

```math
\sum_{j=1}^{k} c_j \left[ B_j'(x_m) + i g'(x_m) B_j(x_m) \right] = f(x_m) \quad \text{for } m = 1, \dots, k
```

## Matrix Formulation and Solution

We can write this linear system in matrix form as:

```math
A \mathbf{c} = \mathbf{f}
```

where,
 $\mathbf{c}$ is the vector of unknown coefficients.
 $\mathbf{f}$ is the vector of amplitude function values $f(x_m)$ at the collocation nodes.
 $A$ is the collocation matrix constructed as $A = (D + \operatorname{diag}(\boldsymbol{\alpha})) B$.
 $B$ is the Chebyshev basis matrix evaluated at the nodes.
 $D$ is the scaled Chebyshev differentiation matrix.
 $\boldsymbol{\alpha}$ is a vector where $\alpha_m = i g'(x_m)$.

Once the matrix $A$ is assembled, the linear system is solved for the coefficients $\mathbf{c}$. Depending on the properties of the phase function and the requested numerical stability, `LevinIntegrals.jl` provides three factorization strategies (solvers) to compute this solution:
1. **QR Factorization** (`QRSolver`): The default solver, offering a good balance of speed and robustness.
2. **LU Factorization** (`LUSolver`): Slightly faster for well-conditioned systems.
3. **Truncated SVD** (`TSVDSolver`): Slowest but maximally stable, effectively regularizing near-singular systems (e.g., when $g'(x) \approx 0$ at stationary phase points).

Finally, after the coefficients $\mathbf{c}$ are obtained, we can evaluate $p(a)$ and $p(b)$ using the basis expansion and compute the final integral $I = p(b) e^{i g(b)} - p(a) e^{i g(a)}$.
