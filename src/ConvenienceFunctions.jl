module ConvenienceFunctions

export
    jacobian_numerical_approximation,
    jacobian_numerical_approximation!,
    zero_small_values,
    zero_small_values!,
    check_symmetric_matrix,
    show_matrix

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

end # module
