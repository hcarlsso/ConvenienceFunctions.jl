using Test
using ConvenienceFunctions
using LinearAlgebra
using Distributions

@testset "Convenience Functions" begin

    @testset "Jacobian numerical approx" begin
        A = randn(4,3)

        f(x) = A*x
        x_0 = randn(3)
        J_0 = jacobian_numerical_approximation(f, x_0, 1.0e-6)

        @test isapprox(J_0, A, atol = 1.0e-6)
    end
    @testset "Test sample covariance inverse" begin
        Nt = 1000
        a = [1;2;3] .+ randn(3, Nt)
        da = a .- mean(a; dims = 2)
        Q_inv_sample = inv(da*da'/(1000 - 1))
        Q_inv = inverse_sample_covariance(a)

        @test isapprox(Q_inv, Q_inv_sample)
    end
    @testset "Print complex matrix" begin
        a = randn(3,3) + randn(3,3)*im
        show_matrix(a)
        print_complex_matrix(a)
    end
    @testset "Points of sphere" begin
        u = points_on_sphere(1000)

        if false
            figure(1);
            clf()
            fig, ax = subplots(3,1, num = 1)
            ax[1].plot(u[1,:], u[2,:], "x")
            ax[1].grid(true)
            ax[2].plot(u[1,:], u[3,:], "x")
            ax[2].grid(true)
            ax[3].plot(u[2,:], u[3,:], "x")
            ax[3].grid(true)
        end
    end
end
