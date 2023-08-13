"Return cis and its inverse"
cis_inv(x::Real) = (z = cis(x); (z, conj(z)))   # inverse of root of unity is conjugate
cis_inv(x::Complex) = (z = cis(x); (z, inv(z)))

istwo(x) = x == (one(x) + one(x))
# Boolean functions that also accept Val arguments
@inline _iszero(x) = iszero(x)
@inline _isone(x)  = isone(x)
@inline _istwo(x)  = istwo(x)

_iszero(::Val{x}) where {x} = iszero(x)
_isone(::Val{x}) where {x}  = isone(x)
_istwo(::Val{x}) where {x}  = istwo(x)

# exponentiation that allows Val arguments
@inline _pow(x, n) = x^n
@generated _pow(x, ::Val{n}) where {n} = :(x ^ $n)

"""
    fourier_contract!(r::AbstractArray{T,N-1}, C::AbstractArray{T,N}, x, [k=1, a=0, shift=0, dim=Val(N)]) where {T,N}

Contract dimension `dim` of array `C` and write it to the array `r`, whose
axes must match `C`'s (excluding dimension `dim`). This function uses the indices in `axes(C,N)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series and must be a `Number`.
The formula for what this routine calculates is:
```math
r_{i_{1},\\dots,i_{N-1}} = \\sum_{i_{N}\\in\\text{axes}(C,N)} C_{i_{1},\\dots,i_{N-1},i_{N}+m+1} (ik (i_{N} + \\text{shift}))^{a} \\exp(ik x (i_{N} + \\text{shift}))
```
"""
@fastmath function fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=inv(oneunit(x)), a=0, shift::Integer=0, ::Val{dim}=Val(N)) where {R,T,N,N_,dim}
    N != N_+1 && throw(ArgumentError("array dimensions incompatible"))
    1 <= dim <= N || throw(ArgumentError("selected dimension to contract is out of bounds"))
    ax = axes(r)
    axC = axes(C)
    (preidx = ax[1:dim-1]) == axC[1:dim-1] || throw(BoundsError("array axes incompatible (check size or indexing)"))
    (sufidx = ax[dim:N_]) == axC[dim+1:N] || throw(BoundsError("array axes incompatible (check size or indexing)"))

    preI = CartesianIndices(preidx)
    sufI = CartesianIndices(sufidx)

    u = k*x                 # unitless, could be complex
    s = size(C, dim); isev = iseven(s)
    c = firstindex(C, dim)  # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    c0 = c+shift            # initial offset index
    z0 = cis(u*c0)          # initial offset phase

    # apply global Fourier multiplier z0*(im*k)^a (k^a could change type)
    z = _iszero(a)  ? z0            :
        _isone(a)   ? z0*im*k       :
        _istwo(a)   ? z0*(im*k)^2   : z0*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type of z)
    b = isev        ? zero(z)   :
        _iszero(a)  ? z         :
        _isone(a)   ? z*c0      :
        _istwo(a)   ? z*c0^2    : z*_pow(c0, a)

    for suffix in sufI, prefix in preI
        @inbounds r[prefix, suffix] = b * C[prefix, c, suffix]
    end

    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = cis_inv(u)   # forward-backward-winding phase steps

    z_, c_ = z, c                  # initial winding phase, index
    _z, _c = isev ? (z*dz_, c-1) : (z, c)

    for n in 1:M            # symmetric winding loop
        _z *= _dz           # forward--winding phase
        z_ *= dz_           # backward-winding phase

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        # winding coefficients
        _b,b_ = _iszero(a)  ? (_z,              z_)                 :
                _isone(a)   ? (_z*(_i+shift),   z_*(i_+shift))      :
                _istwo(a)   ? (_z*(_i+shift)^2, z_*(i_+shift)^2)    : (_z*_pow(_i+shift, a), z_*_pow(i_+shift, a))

        for suffix in sufI, prefix in preI
            @inbounds r[prefix, suffix] += _b * C[prefix, _i, suffix] + b_ * C[prefix, i_, suffix]
        end
    end
    return r
end

"""
    fourier_allocate(C, x, k, a, ::Val{dim})

Allocate an array of the correct type for contracting the Fourier series along axis `dim`.
"""
function fourier_allocate(C::AbstractArray{T,N}, x, k, a, ::Val{dim}) where {T,N,dim}
    ax = axes(C)
    prefix = CartesianIndices(ax[1:dim-1])
    suffix = CartesianIndices(ax[dim+1:N])
    v = view(C, prefix, firstindex(C, dim), suffix)
    y = zero(T)*cis(k*x)
    r = _iszero(a)  ? y             :
        _isone(a)   ? y*im*k        :
        _istwo(a)   ? y*(im*k)^2    : y*_pow(im*k, a)
    return similar(v, typeof(r))
end

"""
    fourier_contract(C::Vector, x, [k=1, a=0, shift=0, dim=Val(N)])

Identical to [`fourier_contract!`](@ref) except that it allocates its output.
"""
function fourier_contract(C::AbstractArray{T,N}, x, k=inv(oneunit(x)), a=0, shift=0, dim=Val(N)) where {T,N}
    # make a copy of the uncontracted dimensions of C while preserving the axes
    r = fourier_allocate(C, x, k, a, dim)
    return fourier_contract!(r, C, x, k, a, shift, dim)
end

"""
    fourier_evaluate(C::AbstractArray{T,N}, x::NTuple{N}, [k=1, a=0, shift=0]) where {T,N}

Evaluates a N-D Fourier series `C`. This function uses the indices in `axes(C)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series and must be a `Number`.
The arguments `x, k, a, shift` must all be tuples of length `N`, the same as the array
dimension. The 1-D formula for what this routine calculates is:
```math
r = \\sum_{i_{\\in\\text{axes}(C,1)} C_{i} (ik (i + \\text{shift}))^{a} \\exp(ik x (i + \\text{shift}))
```
!!! note "Multi-dimensional performance hit"
    This routine is allocation-free, but using it for multidimensional evaluation can be
    slower than allocating because it always computes the Fourier coefficients on the fly.
    Thus, it is typically more efficient to compute the outermost dimensions of the series
    with [`fourier_contract!`](@ref) and then use this routine for the innermost dimension,
    which is faster because it doesn't use inplace operations. [`FourierSeries`](@ref)
    implements this behavior.
"""
@fastmath function fourier_evaluate(C::AbstractArray{T,N}, xs::NTuple{N,Any}, ks::NTuple{N,Any}=map(inv∘oneunit, xs), as::NTuple{N,Any}=ntuple(_ -> Val(0), Val(N)), shifts::NTuple{N,Integer}=ntuple(_ -> 0, Val(N))) where {T,N}
    N == 0 && return C[]

    x_, x = xs[1:N-1], xs[N]
    k_, k = ks[1:N-1], ks[N]
    a_, a = as[1:N-1], as[N]
    shift_, shift = shifts[1:N-1], shifts[N]

    u = k*x                 # unitless, could be complex
    s = size(C, N); isev = iseven(s)
    c = firstindex(C, N)    # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    c0 = c+shift            # initial offset index
    z0 = cis(u*c0)          # initial offset phase

    # apply global Fourier multiplier z0*(im*k)^a (k^a could change type)
    z = _iszero(a)  ? z0            :
        _isone(a)   ? z0*im*k       :
        _istwo(a)   ? z0*(im*k)^2   : z0*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = isev        ? zero(z)   :
        _iszero(a)  ? z         :
        _isone(a)   ? z*c0      :
        _istwo(a)   ? z*c0^2    : z*_pow(c0, a)

    if N == 1
        @inbounds r = b * C[c]  # unroll first loop iteration
    else
        r = b * fourier_evaluate(selectdim(C, N, c), x_, k_, a_, shift_)
    end
    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = cis_inv(u)   # forward-backward-winding phase steps

    z_, c_ = z, c                  # initial winding phase, index
    _z, _c = isev ? (z*dz_, c-1) : (z, c)

    for n in 1:M            # symmetric winding loop
        _z *= _dz           # forward--winding phase
        z_ *= dz_           # backward-winding phase

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        # winding coefficients
        _b,b_ = _iszero(a)  ? (_z,              z_)                 :
                _isone(a)   ? (_z*(_i+shift),   z_*(i_+shift))      :
                _istwo(a)   ? (_z*(_i+shift)^2, z_*(i_+shift)^2)    : (_z*_pow(_i+shift, a), z_*_pow(i_+shift, a))

        if N == 1
            @inbounds r += _b * C[_i] + b_ * C[i_]
        else
            r += _b * fourier_evaluate(selectdim(C, N, _i), x_, k_, a_, shift_) +
                 b_ * fourier_evaluate(selectdim(C, N, i_), x_, k_, a_, shift_)
        end
    end
    return r
end
