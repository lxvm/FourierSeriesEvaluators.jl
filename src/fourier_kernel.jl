"Return cis and its inverse"
cis_inv(x::Real) = (z = cis(x); (z, conj(z)))   # inverse of root of unity is conjugate
cis_inv(x::Complex) = (z = cis(x); (z, inv(z)))

# this version of power should allow constant propagation in more cases than ^
function pow(x_::Number, n::Number)
    x = convert(promote_type(typeof(x_), typeof(n)), x_)
    if iszero(n)
        one(x)
    elseif isone(n)
        x
    elseif n == 2
        x*x
    else
        x^n
    end
end

# this computes pow(op(args...), n), but is lazy about computing op(args...)
function lazypow(n::Number, op, args...)
    iszero(n) ? one(promote_type(Base.promote_op(op, map(typeof, args)...), typeof(n))) : pow(op(args...), n)
end

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
@fastmath function fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=oftype(x,1), a=0, shift::Integer=0, ::Val{dim}=Val(N)) where {R,T,N,N_,dim}
    N != N_+1 && throw(ArgumentError("array dimensions incompatible"))
    R == fourier_type(T, x) || throw(ArgumentError("result array of element type $R needs to store values of type $(fourier_type(T, x))"))

    1 <= dim <= N || throw(ArgumentError("selected dimension to contract is out of bounds"))
    ax = axes(r)
    axC = axes(C)
    (preidx = ax[1:dim-1]) == axC[1:dim-1] || throw(BoundsError("array axes incompatible (check size or indexing)"))
    (sufidx = ax[dim:N_]) == axC[dim+1:N] || throw(BoundsError("array axes incompatible (check size or indexing)"))

    preI = CartesianIndices(preidx)
    sufI = CartesianIndices(sufidx)

    s  = size(C, dim); isev = iseven(s)
    c  = firstindex(C, dim) # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    z  = cis(k*x*(c+shift)) # the initial offset phase
    z *= lazypow(a, *, im, k) # apply global Fourier multiplier (see expression defined above)

    # unroll first loop iteration
    b = !isev * z * lazypow(a, +, c, shift)      # obtain Fourier coefficient
    for suffix in sufI, prefix in preI
        @inbounds r[prefix, suffix] = b * C[prefix, c, suffix]
    end

    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = cis_inv(k*x)          # forward-backward-winding phase steps

    z_ = z                  # initial winding phase
    _z = z * pow(dz_, isev)
    c_ = c                  # initial winding index
    _c = c - isev

    for n in Base.OneTo(M)  # symmetric winding loop
        _z *= _dz           # forward--winding phase
        z_ *= dz_           # backward-winding phase

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        _b = _z * lazypow(a, +, _i, shift)       # forward--winding coefficient
        b_ = z_ * lazypow(a, +, i_, shift)       # backward-winding coefficient

        for suffix in sufI, prefix in preI
            @inbounds r[prefix, suffix] += _b * C[prefix, _i, suffix] + b_ * C[prefix, i_, suffix]
        end
    end
    r
end

"""
    fourier_contract(C::Vector, x, [k=1, a=0, shift=0, dim=Val(N)])

Identical to [`fourier_contract!`](@ref) except that it allocates its output.
"""
function fourier_contract(C::AbstractArray{T,N}, x, k=1, a=0, shift=0, ::Val{dim}=Val(N)) where {T,N,dim}
    # make a copy of the uncontracted dimensions of C while preserving the axes
    ax = axes(C)
    prefix = CartesianIndices(ax[1:dim-1])
    suffix = CartesianIndices(ax[dim+1:N])
    v = view(C, prefix, first(axes(C, dim)), suffix)
    r = similar(v, fourier_type(T,x))
    fourier_contract!(r, C, x, k, a, shift, Val(dim))
end

"""
    fourier_evaluate(C::AbstractVector, x, [k=1, a=0, shift=0])

Evaluates a 1D Fourier series `C`. This function uses the indices in `axes(C,1)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series and must be a `Number`.
The formula for what this routine calculates is:
```math
r = \\sum_{i_{\\in\\text{axes}(C,1)} C_{i} (ik (i + \\text{shift}))^{a} \\exp(ik x (i + \\text{shift}))
```
"""
@fastmath function fourier_evaluate(C::AbstractVector, x, k=oftype(x,1), a=0, shift::Integer=0)
    dim = 1
    s  = size(C, dim); isev = iseven(s)
    c  = firstindex(C, dim) # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    z  = cis(k*x*(c+shift)) # the initial offset phase
    z *= lazypow(a, *, im, k) # apply global Fourier multiplier (see expression defined above)

    b = !isev * z * lazypow(a, +, c, shift)      # obtain Fourier coefficient
    @inbounds r = b * C[c]  # unroll first loop iteration

    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = cis_inv(k*x)          # forward-backward-winding phase steps

    z_ = z                  # initial winding phase
    _z = z * pow(dz_, isev)
    c_ = c                  # initial winding index
    _c = c - isev

    for n in Base.OneTo(M)  # symmetric winding loop
        _z *= _dz           # forward--winding phase
        z_ *= dz_           # backward-winding phase

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        _b = _z * lazypow(a, +, _i, shift)       # forward--winding coefficient
        b_ = z_ * lazypow(a, +, i_, shift)       # backward-winding coefficient

        @inbounds r += _b * C[_i] + b_ * C[i_]
    end
    r
end
