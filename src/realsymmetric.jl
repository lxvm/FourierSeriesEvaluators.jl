"""
    realsymmetric_fourier_evaluate(C::AbstractVector, x, [k=1, a=0])

Evaluates a 1-D RealSymmetric Fourier series `C`, which is a compact representation
of a series with coefficients `C[i] = C[-i]'`. The other arguments are the same
as [`fourier_evaluate`](@ref). The precise formula of what is evaluated is given
by the following, where the initial element ``C_{0}`` is equal to `C[begin]`:
```math
r = 0^a C_{0} + \\sum_{n=1}^{N-1} C_{n} (ik n)^{a} \\exp(ik n x)) + \\text{h.c.}
```
"""
@fastmath function realsymmetric_fourier_evaluate(C::AbstractVector, x, k=inv(oneunit(x)), a=Val(0))
    u = k*x              # unitless
    @assert u isa Real "Real symmetric series can only be evaluated at real arguments"
    s = size(C, 1)
    i = firstindex(C)    # Find the index offset
    e = complex(one(u))  # initial offset phase

    # apply global Fourier multiplier (im*k)^a (k^a could change type)
    z = _iszero(a)  ? e            :
        _isone(a)   ? e*im*k       :
        _istwo(a)   ? e*(im*k)^2   : e*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = _iszero(a)  ? real(z) : 0real(z)
    @inbounds r = _mul(b, C[i]) # unroll first loop iteration
    s == 1 && return r      # fast return for singleton dimensions

    dz = cis(1u)  # forward-backward-winding phase steps
    for n in 1:s-1          # symmetric winding loop
        z *= dz             # forward--winding phase
        i += 1              # forward--winding index
        # winding coefficients
        b = _iszero(a)  ? 2real(z)     :
            _isone(a)   ? 2real(z)*n   :
            _istwo(a)   ? 2real(z)*n^2 : 2real(z)*_pow(n, a)

        @inbounds r += _mul(b, C[i])
    end
    return r
end


"""
    realsymmetric_fourier_contract!(r::AbstractArray, C::AbstractArray, x, k=inv(oneunit(x)), a=0, ::Val{dim}=Val(ndims(C)))

Contracts a single dimension of a multi-dimensional Fourier series, other than the innermost.
Similar to [`fourier_contract!`](@ref) although this function will maintain the conjugate
symmetry of the phase factors in the series.
"""
@fastmath function realsymmetric_fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=inv(oneunit(x)), a=0, ::Val{dim}=Val(N)) where {R,T,N,N_,dim}
    N != N_+1 && throw(ArgumentError("array dimensions incompatible"))
    1 < dim <= N || throw(ArgumentError("selected dimension to contract is out of bounds"))
    ax = axes(r)
    axC = axes(C)
    (preidx = ax[1:dim-1]) == axC[1:dim-1] || throw(BoundsError("array axes incompatible (check size or indexing)"))
    (sufidx = ax[dim:N_]) == axC[dim+1:N] || throw(BoundsError("array axes incompatible (check size or indexing)"))

    preI = CartesianIndices(preidx)
    sufI = CartesianIndices(sufidx)

    u = k*x                 # unitless, could be complex
    @assert u isa Real "RealSymmetric series can only be evaluated at real arguments"
    s = size(C, dim);
    i = firstindex(C, dim)    # Find the index offset
    e = complex(one(u))     # initial offset phase

    # apply global Fourier multiplier z0*(im*k)^a (k^a could change type)
    z = _iszero(a)  ? e            :
        _isone(a)   ? e*im*k       :
        _istwo(a)   ? e*(im*k)^2   : e*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = _iszero(a)  ? real(z) : 0real(z)

    for suffix in sufI, prefix in preI
        @inbounds r[prefix, suffix] = b * C[prefix, i, suffix]
    end

    s == 1 && return r      # fast return for singleton dimensions

    dz = cis(u)     # forward-backward-winding phase steps
    for n in 1:s-1  # symmetric winding loop
        z *= dz     # forward--winding phase
        i += 1      # forward--winding index
        # winding coefficients
        b = _iszero(a)  ? 2real(z)     :
            _isone(a)   ? 2real(z)*n   :
            _istwo(a)   ? 2real(z)*n^2 : 2real(z)*_pow(n, a)

        for suffix in sufI, prefix in preI
            @inbounds r[prefix, suffix] += _mul(b, C[prefix, i, suffix])
        end
    end
    return r
end

"""
    realsymmetric_fourier_allocate(C, x, k, a, ::Val{dim})

Allocate an array of the correct type for contracting the Fourier series along axis `dim`.
"""
function realsymmetric_fourier_allocate(C::AbstractArray{T,N}, x, k, a, ::Val{dim}) where {T,N,dim}
    ax = axes(C)
    prefix = CartesianIndices(ax[1:dim-1])
    suffix = CartesianIndices(ax[dim+1:N])
    v = view(C, prefix, firstindex(C, dim), suffix)
    z = cis(k*x)
    b = _iszero(a)  ? z             :
        _isone(a)   ? z*im*k        :
        _istwo(a)   ? z*(im*k)^2    : z*_pow(im*k, a)
    return similar(v, typeof(_mul(real(b), zero(T))))
end


struct RealSymmetricFourierSeries{N,iip,C<:AbstractArray,A<:NTuple{N,Any},T,F} <: AbstractFourierSeries{N,T,iip}
    c::C
    a::A
    t::NTuple{N,T}
    f::NTuple{N,F}
end

function _check_coeffs_realsym(c::AbstractArray, o::NTuple)
    ax  = axes(c)
    ax_ = ax[1:ndims(c)-length(o)]
    for i in CartesianIndices(ax_)
        for d in ndims(c)-length(o)+1:ndims(c)
            prefix = CartesianIndices(ax[ndims(c)-length(o)+1:d-1])
            suffix = CartesianIndices(ax[d+1:ndims(c)])
            for preI in prefix, sufI in suffix
                _c = @view c[i,preI,CartesianIndices(axes(c,d)),sufI]
                # TODO single, not over, check
                all(c -> c[begin] == c[end] && isreal(c[begin]) && issymmetric(c[begin]), zip(_c, Iterators.reverse(_c))) || return false
            end
        end
    end
    return true
end

"""
    RealSymmetricFourierSeries(f::FourierSeries)

Construct an efficient evaluator for an operator-valued [`FourierSeries`](@ref)
whose coefficients are real symmetric, i.e. ``C_{i} = C_{-i} = C_{i}^{T}``.
Moreover there must be permutation symmetry along axes.
This evaluator guarantees that its output `isreal` and `isymmetric`, and is roughly `2^ndims(f)`
time and memory compared to a `FourierSeries`.

Inplace series are not currently supported
"""
function RealSymmetricFourierSeries(f::FourierSeries)
    isinplace(f) && throw(ArgumentError("inplace series are not supported"))
    _check_coeffs_realsym(f.c, f.o) || throw(ArgumentError("Fourier coefficients do not have RealSymmetric symmetry"))
    _check_coeffs_centered(f.c, f.o) || throw(ArgumentError("Fourier coefficients are not centered"))
    ax = map(ax -> ax[begin+div(length(ax),2):end], axes(f.c))
    c = f.c[ax...] # copy
    return RealSymmetricFourierSeries{ndims(f),isinplace(f),typeof(c),typeof(f.a),eltype(f.t),eltype(f.f)}(c, f.a, f.t, f.f)
end


function evaluate!(c, s::RealSymmetricFourierSeries{1}, x)
    f = freq2rad(s.f[1])
    if isinplace(s)
        error("not implemented")
        # return realsymmetric_fourier_contract!(c, s.c, x, f, s.a[1])
    else
        return realsymmetric_fourier_evaluate(s.c, x, f, s.a[1])
    end
end

function allocate(s::RealSymmetricFourierSeries, x, ::Val{d}) where {d}
    if isinplace(s) || ndims(s) > 1
        M = ndims(s.c)-ndims(s)
        return realsymmetric_fourier_allocate(s.c, x, freq2rad(s.f[d]), s.a[d], Val(M+d))
    elseif isinplace(s) && ndims(s) == 1
        error("not implemented")
        # return realsymmetric_fourier_allocate(s.c, x, freq2rad(s.f[d]), s.a[d])
    else
        return nothing
    end
end

function contract!(c, s::RealSymmetricFourierSeries, x, dim::Val{d}) where {d}
    if d == 1
        error("not implemented")
        # realsymmetric_fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d])
    else
        M = ndims(s.c)-ndims(s)
        realsymmetric_fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d], Val(M+d))
    end
    t = deleteat_(s.t, dim)
    f = deleteat_(s.f, dim)
    a = deleteat_(s.a, dim)
    return RealSymmetricFourierSeries{ndims(s)-1,isinplace(s),typeof(c),typeof(a),eltype(t),eltype(f)}(c, a, t, f)
end

period(s::RealSymmetricFourierSeries) = s.t
frequency(s::RealSymmetricFourierSeries) = s.f

function nextderivative(s::RealSymmetricFourierSeries, dim)
    a = raise_multiplier(s.a, dim)
    return RealSymmetricFourierSeries{ndims(s),isinplace(s),typeof(s.c),typeof(a),eltype(s.t),eltype(s.f)}(s.c, a, s.t, s.f)
end
