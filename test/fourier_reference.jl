using LinearAlgebra

# Definitions for reference Fourier series evaluators
# Arguments:
# - C : The array of Fourier coefficients
# - x : The evaluation point(s)
# - k : The evaluation wavenumber(s)
# - a : The order(s) of derivative of the Fourier series

# sums over only the outermost index of the coefficient array
function ref_contract(C::AbstractArray{T,N}, x::Number, k::Number, a=0, dim=N) where {T,N}
    1 <= dim <= N || error("Choose dim=$dim in 1:$N")
    imk = im*k
    kx = k*x
    R = mapreduce(+, CartesianIndices(C); dims=dim, init=complex(zero(float(T)))) do i
        C[i]*(cis(kx*i.I[dim])*((imk*i.I[dim])^a))
    end
    dropdims(R; dims=dim)
end

# sums over all the indices of the coefficient array
function ref_evaluate(C::AbstractArray{T,N}, x, k, as=0) where {T,N}
    imk = im*k
    kx = k .* x
    sum(CartesianIndices(C), init=complex(zero(float(T)))) do i
        C[i] * (cis(sum(kx .* i.I)) * prod((imk .* i.I) .^ as))
    end
end