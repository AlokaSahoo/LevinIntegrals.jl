using LevinIntegrals
using LinearAlgebra
using Test

@testset "LevinIntegrals.jl" begin

    # ──────────────────────────────────────────────
    # 1. Chebyshev-Lobatto Nodes
    # ──────────────────────────────────────────────
    @testset "chebyshev_lobatto_nodes" begin
        @testset "k = 2 (endpoints only)" begin
            nodes = chebyshev_lobatto_nodes(2)
            @test length(nodes) == 2
            @test nodes[1] ≈ 1.0
            @test nodes[2] ≈ -1.0
        end

        @testset "k = 5" begin
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
            for k in [5, 8, 17, 32]
                nodes = chebyshev_lobatto_nodes(k)
                for i in 1:k
                    @test nodes[i] ≈ -nodes[k + 1 - i] atol=1e-14
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
            for k in [5, 8, 16]
                D = chebyshev_differentiation_matrix(k)
                x = chebyshev_lobatto_nodes(k)
                @test D * x ≈ ones(k) atol=1e-10
            end
        end

        @testset "D * x² = 2x" begin
            for k in [5, 8, 16]
                D = chebyshev_differentiation_matrix(k)
                x = chebyshev_lobatto_nodes(k)
                @test D * (x .^ 2) ≈ 2.0 .* x atol=1e-10
            end
        end

        @testset "D * x³ = 3x²" begin
            for k in [8, 16]
                D = chebyshev_differentiation_matrix(k)
                x = chebyshev_lobatto_nodes(k)
                @test D * (x .^ 3) ≈ 3.0 .* x .^ 2 atol=1e-10
            end
        end

        @testset "D * sin(x) ≈ cos(x)" begin
            D = chebyshev_differentiation_matrix(32)
            x = chebyshev_lobatto_nodes(32)
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
                result = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=32)
                exact = (exp(im * ω) - 1) / (im * ω)
                @test abs(result - exact) < 1e-10
            end
        end

        @testset "∫₀¹ x exp(iωx) dx — linear amplitude" begin
            for ω in [10.0, 100.0]
                result = levin_integrate(x -> x, x -> ω * x, x -> ω, 0.0, 1.0; k=32)
                iω = im * ω
                exact = exp(iω) * (iω - 1) / iω^2 + 1 / iω^2
                @test abs(result - exact) < 1e-10
            end
        end

        @testset "∫₀^π exp(iωx) dx — different interval" begin
            ω = 50.0
            result = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, π; k=32)
            exact = (exp(im * ω * π) - 1) / (im * ω)
            @test abs(result - exact) < 1e-10
        end

        @testset "∫₁³ x² exp(iωx) dx — quadratic amplitude" begin
            ω = 30.0
            iω = im * ω
            a, b = 1.0, 3.0
            exact = (exp(iω * b) * (b^2 / iω - 2b / iω^2 + 2 / iω^3) -
                     exp(iω * a) * (a^2 / iω - 2a / iω^2 + 2 / iω^3))
            result = levin_integrate(x -> x^2, x -> ω * x, x -> ω, a, b; k=32)
            @test abs(result - exact) < 1e-8
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, x -> 1.0, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, x -> 1.0, 0.0, 1.0; k=1)
        end
    end

    # ──────────────────────────────────────────────
    # 6. levin_integrate — with g only (4-arg, numerical g')
    # ──────────────────────────────────────────────
    @testset "levin_integrate (f, g) — spectral derivative" begin

        @testset "∫₀¹ exp(iωx) dx — linear phase" begin
            for ω in [10.0, 100.0, 500.0]
                result = levin_integrate(x -> 1.0, x -> ω * x, 0.0, 1.0; k=32)
                exact = (exp(im * ω) - 1) / (im * ω)
                @test abs(result - exact) < 1e-10
            end
        end

        @testset "∫₀¹ x exp(iωx) dx — linear amplitude, spectral g'" begin
            ω = 50.0
            result = levin_integrate(x -> x, x -> ω * x, 0.0, 1.0; k=32)
            iω = im * ω
            exact = exp(iω) * (iω - 1) / iω^2 + 1 / iω^2
            @test abs(result - exact) < 1e-10
        end

        @testset "Nonlinear phase g(x) = x² — spectral g'" begin
            # ∫₀¹ exp(i x²) dx — Fresnel-type integral
            # Use high-order 5-arg as reference
            ref = levin_integrate(x -> 1.0, x -> x^2, x -> 2x, 0.0, 1.0; k=64)
            result = levin_integrate(x -> 1.0, x -> x^2, 0.0, 1.0; k=64)
            @test abs(result - ref) < 1e-10
        end

        @testset "Agrees with 5-arg dispatch" begin
            ω = 100.0
            r_5arg = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=32)
            r_4arg = levin_integrate(x -> 1.0, x -> ω * x, 0.0, 1.0; k=32)
            @test abs(r_5arg - r_4arg) < 1e-10
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate(x -> 1.0, x -> x, 0.0, 1.0; k=1)
        end
    end

    # ──────────────────────────────────────────────
    # 7. Spectral Convergence
    # ──────────────────────────────────────────────
    @testset "Spectral convergence" begin
        # Use a non-trivial amplitude f(x) = cos(10x) so that small k
        # genuinely cannot resolve it. f(x)=1 is exact at any k.
        ω = 100.0
        # Reference: high-order solve (k=64)
        ref = levin_integrate(x -> cos(10x), x -> ω * x, x -> ω, 0.0, 1.0; k=64)
        errors = Float64[]
        for k in [4, 8, 16, 32]
            result = levin_integrate(x -> cos(10x), x -> ω * x, x -> ω, 0.0, 1.0; k=k)
            push!(errors, abs(result - ref))
        end
        # Errors should decrease as k increases
        @test errors[1] > errors[3]
        @test errors[2] > errors[4]
        # At k=32, should be very close to the k=64 reference
        @test errors[4] < 1e-10
    end

    # ──────────────────────────────────────────────
    # 8. Composite (fixed sub-intervals) Levin Integration
    # ──────────────────────────────────────────────
    @testset "levin_integrate composite (n panels)" begin

        @testset "n=1 matches single-panel result" begin
            ω = 100.0
            r1 = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=16, n=1)
            r0 = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=16)
            @test r1 ≈ r0
        end

        @testset "Composite improves accuracy for rapidly-varying amplitude" begin
            ω = 50.0
            a, b = 0.0, 2.0
            f = x -> exp(5x)
            # Reference: high-order single panel
            ref = levin_integrate(f, x -> ω * x, x -> ω, a, b; k=64)
            # Low-order single panel
            low = levin_integrate(f, x -> ω * x, x -> ω, a, b; k=8)
            # Low-order with multiple panels
            composite = levin_integrate(f, x -> ω * x, x -> ω, a, b; k=8, n=4)
            @test abs(composite - ref) < abs(low - ref)
        end

        @testset "Composite with exact result" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=16, n=4)
            @test abs(result - exact) < 1e-10
        end

        @testset "Composite with numerical g' (4-arg)" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result = levin_integrate(x -> 1.0, x -> ω * x, 0.0, 1.0; k=16, n=4)
            @test abs(result - exact) < 1e-10
        end

        @testset "Argument errors for n" begin
            @test_throws ArgumentError levin_integrate(
                x -> 1.0, x -> x, x -> 1.0, 0.0, 1.0; n=0)
        end
    end

    # ──────────────────────────────────────────────
    # 9. Adaptive Levin Integration
    # ──────────────────────────────────────────────
    @testset "levin_integrate_adaptive" begin

        @testset "Agrees with fixed-order on smooth integrand" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result_fixed    = levin_integrate(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0; k=16)
            result_adaptive = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                        k=16, atol=1e-12)
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
                                               k=16, atol=1e-12)
            @test abs(result - exact) < 1e-10
        end

        @testset "Handles rapidly-varying amplitude via refinement" begin
            ω = 50.0
            a, b = 0.0, 2.0
            f = x -> exp(5x)
            # Reference: high-order fixed solve
            ref = levin_integrate(f, x -> ω * x, x -> ω, a, b; k=64)
            # Low-order fixed might be inaccurate
            low = levin_integrate(f, x -> ω * x, x -> ω, a, b; k=8)
            # Adaptive with k=8 per panel should refine and match reference
            adpt = levin_integrate_adaptive(f, x -> ω * x, x -> ω, a, b;
                                             k=8, atol=1e-10)
            @test abs(adpt - ref) < abs(low - ref)
        end

        @testset "Adaptive with numerical g' (4-arg)" begin
            ω = 100.0
            exact = (exp(im * ω) - 1) / (im * ω)
            result = levin_integrate_adaptive(x -> 1.0, x -> ω * x, 0.0, 1.0;
                                               k=16, atol=1e-12)
            @test abs(result - exact) < 1e-10
        end

        @testset "Both adaptive dispatches agree" begin
            ω = 100.0
            r_5arg = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                               k=16, atol=1e-12)
            r_4arg = levin_integrate_adaptive(x -> 1.0, x -> ω * x, 0.0, 1.0;
                                               k=16, atol=1e-12)
            @test abs(r_5arg - r_4arg) < 1e-10
        end

        @testset "maxdepth is respected" begin
            ω = 100.0
            result = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                               k=8, maxdepth=0)
            @test isfinite(abs(result))
        end

        @testset "Tighter tolerance gives better accuracy" begin
            ω = 200.0
            exact = (exp(im * ω) - 1) / (im * ω)
            r_loose = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                k=4, atol=1e-4)
            r_tight = levin_integrate_adaptive(x -> 1.0, x -> ω * x, x -> ω, 0.0, 1.0;
                                                k=4, atol=1e-12)
            @test abs(r_tight - exact) ≤ abs(r_loose - exact) + eps()
        end

        @testset "Argument errors" begin
            @test_throws ArgumentError levin_integrate_adaptive(
                x -> 1.0, x -> x, x -> 1.0, 1.0, 0.0)
            @test_throws ArgumentError levin_integrate_adaptive(
                x -> 1.0, x -> x, 1.0, 0.0)
        end
    end

    # ──────────────────────────────────────────────
    # 10. Solver Strategy Types
    # ──────────────────────────────────────────────
    @testset "Solver strategies" begin

        ω     = 100.0
        exact = (exp(im * ω) - 1) / (im * ω)
        f     = x -> 1.0
        g     = x -> ω * x
        gp    = x -> ω

        @testset "Solver type hierarchy" begin
            @test QRSolver()    isa LevinSolver
            @test LUSolver()    isa LevinSolver
            @test TSVDSolver()  isa LevinSolver
            @test TSVDSolver(1e-12) isa LevinSolver
            @test TSVDSolver().tol  == 1e-14
            @test TSVDSolver(1e-10).tol == 1e-10
        end

        @testset "All solvers agree — levin_integrate (5-arg)" begin
            r_qr   = levin_integrate(f, g, gp, 0.0, 1.0; k=32, solver=QRSolver())
            r_lu   = levin_integrate(f, g, gp, 0.0, 1.0; k=32, solver=LUSolver())
            r_tsvd = levin_integrate(f, g, gp, 0.0, 1.0; k=32, solver=TSVDSolver())

            @test abs(r_qr   - exact) < 1e-10
            @test abs(r_lu   - exact) < 1e-10
            @test abs(r_tsvd - exact) < 1e-10
            @test abs(r_qr - r_lu)   < 1e-10
            @test abs(r_qr - r_tsvd) < 1e-10
        end

        @testset "All solvers agree — levin_integrate (4-arg, spectral g')" begin
            r_qr   = levin_integrate(f, g, 0.0, 1.0; k=32, solver=QRSolver())
            r_lu   = levin_integrate(f, g, 0.0, 1.0; k=32, solver=LUSolver())
            r_tsvd = levin_integrate(f, g, 0.0, 1.0; k=32, solver=TSVDSolver())

            @test abs(r_qr   - exact) < 1e-10
            @test abs(r_lu   - exact) < 1e-10
            @test abs(r_tsvd - exact) < 1e-10
            @test abs(r_qr - r_lu)   < 1e-10
            @test abs(r_qr - r_tsvd) < 1e-10
        end

        @testset "All solvers agree — levin_integrate_adaptive (5-arg)" begin
            r_qr   = levin_integrate_adaptive(f, g, gp, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=QRSolver())
            r_lu   = levin_integrate_adaptive(f, g, gp, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=LUSolver())
            r_tsvd = levin_integrate_adaptive(f, g, gp, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=TSVDSolver())

            @test abs(r_qr   - exact) < 1e-10
            @test abs(r_lu   - exact) < 1e-10
            @test abs(r_tsvd - exact) < 1e-10
            @test abs(r_qr - r_lu)   < 1e-10
            @test abs(r_qr - r_tsvd) < 1e-10
        end

        @testset "All solvers agree — levin_integrate_adaptive (4-arg, spectral g')" begin
            r_qr   = levin_integrate_adaptive(f, g, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=QRSolver())
            r_lu   = levin_integrate_adaptive(f, g, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=LUSolver())
            r_tsvd = levin_integrate_adaptive(f, g, 0.0, 1.0; k=16, atol=1e-12,
                                               solver=TSVDSolver())

            @test abs(r_qr   - exact) < 1e-10
            @test abs(r_lu   - exact) < 1e-10
            @test abs(r_tsvd - exact) < 1e-10
            @test abs(r_qr - r_lu)   < 1e-10
            @test abs(r_qr - r_tsvd) < 1e-10
        end

        @testset "TSVDSolver accepts custom tolerance" begin
            r1 = levin_integrate(f, g, gp, 0.0, 1.0; k=32, solver=TSVDSolver(1e-14))
            r2 = levin_integrate(f, g, gp, 0.0, 1.0; k=32, solver=TSVDSolver(1e-10))
            # Both should still give accurate results for a well-conditioned system
            @test abs(r1 - exact) < 1e-10
            @test abs(r2 - exact) < 1e-10
        end

        @testset "Backward compatibility — no solver kwarg" begin
            # All existing call patterns must work unchanged
            r1 = levin_integrate(f, g, gp, 0.0, 1.0; k=32)
            r2 = levin_integrate(f, g, 0.0, 1.0; k=32)
            r3 = levin_integrate_adaptive(f, g, gp, 0.0, 1.0; k=16, atol=1e-12)
            r4 = levin_integrate_adaptive(f, g, 0.0, 1.0; k=16, atol=1e-12)

            @test abs(r1 - exact) < 1e-10
            @test abs(r2 - exact) < 1e-10
            @test abs(r3 - exact) < 1e-10
            @test abs(r4 - exact) < 1e-10
        end

        @testset "Composite Levin with non-default solver" begin
            r_lu   = levin_integrate(f, g, gp, 0.0, 1.0; k=8, n=4, solver=LUSolver())
            r_tsvd = levin_integrate(f, g, gp, 0.0, 1.0; k=8, n=4, solver=TSVDSolver())
            @test abs(r_lu   - exact) < 1e-10
            @test abs(r_tsvd - exact) < 1e-10
        end

    end

end
