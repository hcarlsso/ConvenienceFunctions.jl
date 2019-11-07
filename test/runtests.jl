using Test
using ConvenienceFunctions
using LinearAlgebra

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
end
