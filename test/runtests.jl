using LevinIntegrals
using LinearAlgebra
using Test

@testset "LevinIntegrals.jl" begin

    # ──────────────────────────────────────────────
    # 1. Chebyshev-Lobatto Nodes
    # ──────────────────────────────────────────────
    @testset "chebyshev_lobatto_nodes" begin
        @testset "n = 2 (endpoints only)" begin
            nodes = chebyshev_lobatto_nodes(2)
            @test length(nodes) == 2
            @test nodes[1] ≈ 1.0
            @test nodes[2] ≈ -1.0
        end

        @testset "n = 5" begin
            nodes = chebyshev_lobatto_nodes(5)
            @test length(nodes) == 5
            @test nodes[1] ≈ 1.0
            @test nodes[end] ≈ -1.0
            @test nodes[3] ≈ 0.0 atol=1e-15
        end

        @testset "Descending order" begin
            nodes = chebyshev_lobatto_nodes(10)
            @test issorted(nodes, rev=true)
        end

        @testset "Symmetry" begin
            for n in [5, 8, 17, 32]
                nodes = chebyshev_lobatto_nodes(n)
                for i in 1:n
                    @test nodes[i] ≈ -nodes[n + 1 - i] atol=1e-14
                end
            end
        end

        @testset "All nodes in [-1, 1]" begin
            nodes = chebyshev_lobatto_nodes(50)
            @test all(-1.0 ≤ x ≤ 1.0 for x in nodes)
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError chebyshev_lobatto_nodes(1)
            @test_throws ArgumentError chebyshev_lobatto_nodes(0)
        end
    end

    # ──────────────────────────────────────────────
    # 2. Node Mapping
    # ──────────────────────────────────────────────
    @testset "map_nodes" begin
        nodes = chebyshev_lobatto_nodes(5)

        @testset "Map to [0, 1]" begin
            mapped = LevinIntegrals.map_nodes(nodes, 0.0, 1.0)
            @test mapped[1] ≈ 1.0
            @test mapped[end] ≈ 0.0
        end

        @testset "Map to [2, 6]" begin
            mapped = LevinIntegrals.map_nodes(nodes, 2.0, 6.0)
            @test mapped[1] ≈ 6.0
            @test mapped[end] ≈ 2.0
            @test mapped[3] ≈ 4.0 atol=1e-14
        end

        @testset "Identity mapping [-1, 1]" begin
            mapped = LevinIntegrals.map_nodes(nodes, -1.0, 1.0)
            @test mapped ≈ nodes
        end
    end

    # ──────────────────────────────────────────────
    # 3. Chebyshev Basis Matrix
    # ──────────────────────────────────────────────
    @testset "chebyshev_basis_matrix" begin
        nodes = chebyshev_lobatto_nodes(5)

        @testset "T₀ = 1 everywhere" begin
            B = LevinIntegrals.chebyshev_basis_matrix(nodes, 5)
            @test all(B[:, 1] .≈ 1.0)
        end

        @testset "T₁ = x" begin
            B = LevinIntegrals.chebyshev_basis_matrix(nodes, 5)
            @test B[:, 2] ≈ nodes
        end

        @testset "T₂(x) = 2x² - 1" begin
            B = LevinIntegrals.chebyshev_basis_matrix(nodes, 5)
            @test B[:, 3] ≈ 2.0 .* nodes .^ 2 .- 1.0
        end

        @testset "T₃(x) = 4x³ - 3x" begin
            B = LevinIntegrals.chebyshev_basis_matrix(nodes, 5)
            @test B[:, 4] ≈ 4.0 .* nodes .^ 3 .- 3.0 .* nodes
        end

        @testset "Square matrix dimensions" begin
            B = LevinIntegrals.chebyshev_basis_matrix(nodes, 5)
            @test size(B) == (5, 5)
        end
    end

    # ──────────────────────────────────────────────
    # 4. Differentiation Matrix
    # ──────────────────────────────────────────────
    @testset "chebyshev_differentiation_matrix" begin
        @testset "D * x = 1" begin
            for n in [5, 8, 16]
                D = chebyshev_differentiation_matrix(n)
                x = chebyshev_lobatto_nodes(n)
                @test D * x ≈ ones(n) atol=1e-10
            end
        end

        @testset "D * x² = 2x" begin
            for n in [5, 8, 16]
                D = chebyshev_differentiation_matrix(n)
                x = chebyshev_lobatto_nodes(n)
                @test D * (x .^ 2) ≈ 2.0 .* x atol=1e-10
            end
        end

        @testset "D * x³ = 3x²" begin
            for n in [8, 16]
                D = chebyshev_differentiation_matrix(n)
                x = chebyshev_lobatto_nodes(n)
                @test D * (x .^ 3) ≈ 3.0 .* x .^ 2 atol=1e-10
            end
        end

        @testset "D * sin(x) ≈ cos(x)" begin
            n = 32
            D = chebyshev_differentiation_matrix(n)
            x = chebyshev_lobatto_nodes(n)
            @test D * sin.(x) ≈ cos.(x) atol=1e-10
        end

        @testset "Row sums are zero" begin
            D = chebyshev_differentiation_matrix(8)
            row_sums = sum(D, dims=2)
            @test all(abs.(row_sums) .< 1e-12)
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError chebyshev_differentiation_matrix(1)
        end
    end

    # ──────────────────────────────────────────────
    # 5. levin_integrate — with g and g_prime (5-arg)
    # ──────────────────────────────────────────────
    @testset "levin_integrate (f, g, g_prime)" begin

        @testset "∫₀¹ exp(iωx) dx — constant amplitude" begin
            for ω in [10.0, 50.0, 100.0, 500.0]
                result = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; n=32)
                exact = (exp(im * ω) - 1) / (im * ω)
                @test abs(result - exact) < 1e-10
            end
        end

        @testset "∫₀¹ x exp(iωx) dx — linear amplitude" begin
            for ω in [10.0, 100.0]
                result = levin_integrate(x -> x, x -> ω * x, x -> ω, 0.0, 1.0; n=32)
                iω = im * ω
                exact = exp(iω) * (iω - 1) / iω^2 + 1 / iω^2
                @test abs(result - exact) < 1e-10
            end
        end

        @testset "∫₀^π exp(iωx) dx — different interval" begin
            ω = 50.0
            result = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, π; n=32)
            exact = (exp(im * ω * π) - 1) / (im * ω)
            @test abs(result - exact) < 1e-10
        end

        @testset "∫₁³ x² exp(iωx) dx — quadratic amplitude" begin
            ω = 30.0
            iω = im * ω
            a, b = 1.0, 3.0
            exact = (exp(iω * b) * (b^2 / iω - 2b / iω^2 + 2 / iω^3) -
                     exp(iω * a) * (a^2 / iω - 2a / iω^2 + 2 / iω^3))
            result = levin_integrate(x -> x^2, x -> ω * x, x -> ω, a, b; n=32)
            @test abs(result - exact) < 1e-8
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, x -> 1.0, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, x -> 1.0, 0.0, 1.0; n=1)
        end
    end

    # ──────────────────────────────────────────────
    # 6. levin_integrate — with g only (4-arg, numerical g')
    # ──────────────────────────────────────────────
    @testset "levin_integrate (f, g) — numerical derivative" begin

        @testset "∫₀¹ exp(iωx) dx — linear phase" begin
            for ω in [10.0, 100.0, 500.0]
                result = levin_integrate(x -> 1.0, x -> ω * x, 0.0, 1.0; n=32)
                exact = (exp(im * ω) - 1) / (im * ω)
                # Slightly relaxed tolerance due to finite-difference g'
                @test abs(result - exact) < 1e-8
            end
        end

        @testset "∫₀¹ x exp(iωx) dx — linear amplitude, numerical g'" begin
            ω = 50.0
            result = levin_integrate(x -> x, x -> ω * x, 0.0, 1.0; n=32)
            iω = im * ω
            exact = exp(iω) * (iω - 1) / iω^2 + 1 / iω^2
            @test abs(result - exact) < 1e-8
        end

        @testset "Nonlinear phase g(x) = x² — numerical g'" begin
            # ∫₀¹ exp(i x²) dx — Fresnel-type integral
            # Use high-order 5-arg as reference
            ref = levin_integrate(x -> 1.0, x -> x^2, x -> 2x, 0.0, 1.0; n=64)
            result = levin_integrate(x -> 1.0, x -> x^2, 0.0, 1.0; n=64)
            @test abs(result - ref) < 1e-6
        end

        @testset "Agrees with 5-arg dispatch" begin
            ω = 100.0
            r_5arg = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; n=32)
            r_4arg = levin_integrate(x -> 1.0, x -> ω * x, 0.0, 1.0; n=32)
            @test abs(r_5arg - r_4arg) < 1e-8
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, 0.0, 1.0; n=1)
        end
    end

    # ──────────────────────────────────────────────
    # 7. Spectral Convergence
    # ──────────────────────────────────────────────
    @testset "Spectral convergence" begin
        # Use a non-trivial amplitude f(x) = cos(10x) so that small n
        # genuinely cannot resolve it. f(x)=1 is exact at any n.
        ω = 100.0
        # Reference: high-order solve (n=64)
        ref = levin_integrate(x -> cos(10x), x -> ω * x, x -> ω, 0.0, 1.0; n=64)
        errors = Float64[]
        for n in [4, 8, 16, 32]
            result = levin_integrate(x -> cos(10x), x -> ω * x, x -> ω, 0.0, 1.0; n=n)
            push!(errors, abs(result - ref))
        end
        # Errors should decrease as n increases
        @test errors[1] > errors[3]
        @test errors[2] > errors[4]
        # At n=32, should be very close to the n=64 reference
        @test errors[4] < 1e-10
    end

    # ──────────────────────────────────────────────
    # 8. Adaptive Levin Integration
    # ──────────────────────────────────────────────
    @testset "levin_integrate_adaptive" begin

        @testset "Agrees with fixed-order on smooth integrand" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result_fixed    = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; n=16)
            result_adaptive = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                        n=16, atol=1e-12)
            @test abs(result_adaptive - exact) < 1e-10
            @test abs(result_adaptive - result_fixed) < 1e-10
        end

        @testset "Smooth quadratic amplitude" begin
            ω = 50.0
            iω = im * ω
            a, b = 0.0, 2.0
            exact = (exp(iω * b) * (b^2 / iω - 2b / iω^2 + 2 / iω^3) -
                     exp(iω * a) * (a^2 / iω - 2a / iω^2 + 2 / iω^3))
            result = levin_integrate_adaptive(x -> x^2, x -> ω * x, x -> ω, a, b;
                                               n=16, atol=1e-12)
            @test abs(result - exact) < 1e-10
        end

        @testset "Handles rapidly-varying amplitude via refinement" begin
            ω = 50.0
            a, b = 0.0, 2.0
            f = x -> exp(5x)
            # Reference: high-order fixed solve
            ref = levin_integrate(f, x -> ω * x, x -> ω, a, b; n=64)
            # Low-order fixed might be inaccurate
            low = levin_integrate(f, x -> ω * x, x -> ω, a, b; n=8)
            # Adaptive with n=8 per panel should refine and match reference
            adpt = levin_integrate_adaptive(f, x -> ω * x, x -> ω, a, b;
                                             n=8, atol=1e-10)
            @test abs(adpt - ref) < abs(low - ref)
        end

        @testset "Adaptive with numerical g' (4-arg)" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result = levin_integrate_adaptive(x -> 1.0, x -> ω * x, 0.0, 1.0;
                                               n=16, atol=1e-10)
            @test abs(result - exact) < 1e-8
        end

        @testset "Both adaptive dispatches agree" begin
            ω = 100.0
            r_5arg = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                               n=16, atol=1e-12)
            r_4arg = levin_integrate_adaptive(x -> 1.0, x -> ω * x, 0.0, 1.0;
                                               n=16, atol=1e-12)
            @test abs(r_5arg - r_4arg) < 1e-8
        end

        @testset "maxdepth is respected" begin
            ω = 100.0
            result = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                               n=8, maxdepth=0)
            @test isfinite(abs(result))
        end

        @testset "Tighter tolerance gives better accuracy" begin
            ω = 200.0
            exact = (exp(im * ω) - 1) / (im * ω)
            r_loose = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                n=4, atol=1e-4)
            r_tight = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                n=4, atol=1e-12)
            @test abs(r_tight - exact) ≤ abs(r_loose - exact) + eps()
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate_adaptive(
                x -> 1.0, x -> x, x -> 1.0, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate_adaptive(
                x -> 1.0, x -> x, 1.0, 0.0)
        end
    end

end
