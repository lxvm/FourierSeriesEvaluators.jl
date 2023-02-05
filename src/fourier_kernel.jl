
"""
    fourier_kernel!(r::Array{T,N-1}, C::Array{T,N}, x, ξ, [::Val{a}=Val{0}()]) where {T,N,a}

Contract the outermost index of array `C` and write it to the array `r`. Assumes
the size of the outermost dimension of `C` is `2m+1` and sums the coefficients
```math
r_{i_{1},\\dots,i_{N-1}} = \\sum_{i_{N}=-m}^{m} C_{i_{1},\\dots,i_{N-1},i_{N}+m+1} (i2\\pi\\xi i_{N})^{a} \\exp(i2\\pi\\xi x i_{N})
```
Hence this represents evaluation of a Fourier series with `m` modes. The
parameter `a` represents the order of derivative of the Fourier series.
"""
@generated function fourier_kernel!(r::Array{T,N_}, C::Array{T,N}, x, ξ, ::Val{a}=Val{0}()) where {T,N,N_,a}
    N != N_+1 && return :(error("array dimensions incompatible"))
    if a == 0
        fundamental = :(Base.Cartesian.@nref $N C d -> d == $N ? m+1 : i_d)
        c = :(z); c̄ = :(conj(z))
    elseif a == 1
        fundamental = :(zero($T))
        c = :(im*2π*ξ*n*z); c̄ = :(conj(c))
    else
        f₀ = 0^a
        fundamental = :(fill($f₀, $T))
        c = :(((im*2pi*ξ*n)^$a)*z); c̄ = :(((-im*2pi*ξ*n)^$a)*conj(z))
    end
    quote
        size(r) == size(C)[1:$N_] || error("array sizes incompatible")
        s = size(C,$N)
        isodd(s) || return error("expected an array with an odd number of coefficients")
        m = div(s,2)
        @inbounds Base.Cartesian.@nloops $N_ i r begin
            (Base.Cartesian.@nref $N_ r i) = $fundamental
        end
        z₀ = cispi(2ξ*x)
        z = one(z₀)
        for n in Base.OneTo(m)
            z *= z₀
            c  = $c
            c̄  = $c̄
            @inbounds Base.Cartesian.@nloops $N_ i r begin
                (Base.Cartesian.@nref $N_ r i) += c*(Base.Cartesian.@nref $N C d -> d == $N ? n+m+1 : i_d) + c̄*(Base.Cartesian.@nref $N C d -> d == $N ? -n+m+1 : i_d)
            end
        end
    end
end

"""
    fourier_kernel(C::Vector, x, ξ, [::Val{a}=Val{0}()])

A version of `fourier_kernel!` for 1D Fourier series evaluation that is not in
place, but allocates an output array. This is usually faster for series whose
element type is a StaticArray for integrals that don't need to reuse the data.
"""
@generated function fourier_kernel(C::Vector{T}, x, ξ, ::Val{a}=Val{0}()) where {T,a}
    if a == 0
        fundamental = :(C[m+1])
        c = :(z); c̄ = :(conj(z))
    elseif a == 1
        fundamental = :(zero($T))
        c = :(im*2π*ξ*n*z); c̄ = :(conj(c))
    else
        f₀ = 0^a
        fundamental = :(fill($f₀, $T))
        c = :(((im*2pi*ξ*n)^$a)*z); c̄ = :(((-im*2pi*ξ*n)^$a)*conj(z))
    end
    quote
        s = size(C,1)
        isodd(s) || return error("expected an array with an odd number of coefficients")
        m = div(s,2)
        @inbounds r = $fundamental
        z₀ = cispi(2ξ*x)
        z = one(z₀)
        @inbounds for n in Base.OneTo(m)
            z *= z₀
            c  = $c
            c̄  = $c̄
            r += c*C[n+m+1] + c̄*C[-n+m+1]
        end
        r
    end
end
