module ConvenienceFunctions

using LinearAlgebra
using Distributions
using Formatting


export
    inverse_sample_covariance,
    points_on_sphere,
    jacobian_numerical_approximation,
    jacobian_numerical_approximation!,
    gradient_numerical_approximation,
    gradient_numerical_approximation!,
    zero_small_values,
    zero_small_values!,
    check_symmetric_matrix,
    show_matrix,
    print_complex_matrix


function print_complex_matrix(data::AbstractMatrix; cols = 4,
                              fmt::String = "%.4f", sep::String = "\t")

    f = generate_formatter( fmt )
    ncols = size(data,2)
    cols_full = fld(ncols,cols)
    last_row = mod(ncols,cols)
    inds_cols = vcat(
        [(1:cols) .+ cols*k for k = 0:cols_full-1],
        [(1:last_row) .+ cols_full*cols]
    )
    for inds in inds_cols
        println("Cols $(inds[1]) to $(inds[end])\n")
        for i = 1:size(data,1)
            for j in inds
                val = data[i,j]
                if typeof(val) <: Complex
                    print(f(real(val)), " ")
                    print(f(imag(val) ),"i")
                else
                    f(val)
                end
                print(sep)
            end
            println("")
        end
    end
end
"""
     jacobian_numerical_approximation(f, x, myeps = 1.0e-6)

Evaluate the jacobian of function f at x with steplength myeps.
"""
function jacobian_numerical_approximation(func, x, myeps = 1.0e-6)
    n = length(x)
    fx = func(x)
    J = Array{eltype(x)}(undef, length(fx), length(x))
    jacobian_numerical_approximation!(J, func, x, myeps)
    return J
end
function jacobian_numerical_approximation!(J, func, x, myeps = 1.0e-6)

    n = length(x)
    fx = func(x)
    xperturb = copy(x)
    for i=1:n
        xperturb[i] = xperturb[i] + myeps
        J[:,i] = (func(xperturb) - fx)./myeps
        xperturb[i] = x[i]
    end
    return J
end
"""
    gradient_numerical_approximation(f, x, myeps = 1.0e-6)

Evaluate the gradient (one dimensional Jacobian) of function f at x with
steplength myeps.
"""
function gradient_numerical_approximation(func, x, myeps = 1.0e-6)
    n = length(x)
    G = Array{eltype(x)}(undef, length(x))
    gradient_numerical_approximation!(G, func, x, myeps)
    return G
end
"""
    gradient_numerical_approximation(f, x, myeps = 1.0e-6)

Evaluate the gradient (one dimensional Jacobian) of function f at x with
steplength myeps.
"""
function gradient_numerical_approximation!(G, func, x, myeps = 1.0e-6)

    n = length(x)
    fx = func(x)
    xperturb = copy(x)
    for i=1:n
        xperturb[i] = xperturb[i] + myeps
        G[i] = (func(xperturb) - fx)./myeps
        xperturb[i] = x[i]
    end
    return G
end
function check_symmetric_matrix(A)
    if norm(A - transpose(A)) < 1e-7
        return Symmetric(A)
    else
        return A
    end
end
function show_matrix(x...)
    for x_i in x
        if typeof(x_i) <: String
            println(x_i)
        else
            show_matrix(x_i)
        end
    end
end
"""
    show_matrix(x)

Wrapper for `show(stdout, "text/plain", x)`
"""
function show_matrix(x::AbstractArray)
    show(stdout, "text/plain", x)
    println("\n")
end

function zero_small_values(x, margin = 10.0)
    x_copy = deepcopy(x)
    zero_small_values!(x_copy, margin)
    return x_copy
end

function zero_small_values!(x, margin = 10.0)
    x[abs.(x) .< margin*eps(eltype(x))] .= zero(eltype(x))
    return x
end
function points_on_sphere(num_pts::Integer, T::DataType=Float64)
    u = zeros(T, 3, num_pts)
    points_on_sphere!(u, num_pts)
    return u
end
"""
    points_on_sphere!(u, num_pts)

Place points on sphere according to sunflower algorithm:
https://stackoverflow.com/a/44164075/2082968

"""
function points_on_sphere!(u::AbstractMatrix, num_pts::Integer)

    T = eltype(u)
    indices = convert.(T, collect(0:num_pts-1))

    phi = acos.(1.0 .- 2.0*indices/convert(T, num_pts))
    theta = pi * (1.0 + sqrt(5)) * indices

    u[1,:] = cos.(theta) .* sin.(phi) # x
    u[2,:] = sin.(theta) .* sin.(phi) # y
    u[3,:] = cos.(phi) # z
    return u
end
"""
    inverse_sample_covariance(y)

Inverse sample covariance over time-dimension assumed to be dim = 2.
"""
function inverse_sample_covariance(y::AbstractMatrix)
    dy =  y .- mean(y; dims = 2)
    F = qr(dy')
    R_inv = inv(F.R)
    Q_inv = R_inv*(R_inv')*(size(y,2) - 1)
    return Q_inv
end

end # module
