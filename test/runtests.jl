using Test
using ConvenienceFunctions

@testset "Convenience Functions" begin

    A = randn(4,3)

    f(x) = A*x
    x_0 = randn(3)
    J_0 = jacobian_numerical_approximation(f, x_0, 1.0e-6)

    @test isapprox(J_0, A, atol = 1.0e-6)
end
