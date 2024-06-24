# here we make use of the fact that for coefficients that satisfy A[i] = A[-i]' produce
# hermitian-valued Fourier series since z*A[i] + z'*A[-i] = (z*A[i]) + (z*A[i])'

# One can also define a series having symmetric coefficients satisfying A[i] = A[-i] = transpose(A[i]).
# This alone should guarantee that the output is also symmetric, but is most likely not
# worth implementing separately because the evaluation of the series one dimension at a time
# only preserves the Hermitian property of the intermediate coefficients.

# If the coefficients are real symmetric and satisfy A[i] = A[-i] along each axis, then only
# half of the coefficients along each dimension are needed to represent the series.
# This case is not very general, so we do not specialize on it to attain a roughly 2x
# speedup, although the routines below should preserve the real symmetric property numerically.

_hermitianpart(a::Number) = real(a)
_hermitianpart(A::AbstractMatrix) = Hermitian(A)

"""
    hermitian_fourier_evaluate(C::AbstractVector, x, [k=1, a=0])

Evaluates a 1-D Hermitian Fourier series `C`, which is a compact representation
of a series with coefficients `C[i] = C[-i]'`. The other arguments are the same
as [`fourier_evaluate`](@ref). The precise formula of what is evaluated is given
by the following, where the initial element ``C_{0}`` is equal to `C[begin]`:
```math
r = 0^a C_{0} + \\sum_{n=1}^{N-1} C_{n} (ik n)^{a} \\exp(ik n x)) + \\text{h.c.}
```
"""
@fastmath function hermitian_fourier_evaluate(C::AbstractVector, x, k=inv(oneunit(x)), a=Val(0))
    u = k*x              # unitless
    @assert u isa Real "Hermitian series can only be evaluated at real arguments"
    s = size(C, 1)
    i = firstindex(C)    # Find the index offset
    e = complex(one(u))  # initial offset phase

    # apply global Fourier multiplier (im*k)^a (k^a could change type)
    z = _iszero(a)  ? e            :
        _isone(a)   ? e*im*k       :
        _istwo(a)   ? e*(im*k)^2   : e*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = _iszero(a)  ? z : 0z
    @inbounds r = _hermitianpart(_mul(b, C[i])) # unroll first loop iteration
    s == 1 && return r      # fast return for singleton dimensions

    dz = cis(1u)  # forward-backward-winding phase steps
    for n in 1:s-1          # symmetric winding loop
        z *= dz             # forward--winding phase
        i += 1              # forward--winding index
        # winding coefficients
        b = _iszero(a)  ? z     :
            _isone(a)   ? z*n   :
            _istwo(a)   ? z*n^2 : z*_pow(n, a)

        @inbounds B = _mul(b, C[i])
        r += _hermitianpart(B) + _hermitianpart(B')
    end
    return r
end

"""
    hermitian_fourier_contract!(r::AbstractArray, C::AbstractArray, x, k=inv(oneunit(x)), a=0, ::Val{dim}=Val(ndims(C)))

Contracts a single dimension of a multi-dimensional Fourier series, other than the innermost.
Similar to [`fourier_contract!`](@ref) although this function will maintain the conjugate
symmetry of the phase factors in the series.
"""
@fastmath function hermitian_fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=inv(oneunit(x)), a=0, ::Val{dim}=Val(N)) where {R,T,N,N_,dim}
    N != N_+1 && throw(ArgumentError("array dimensions incompatible"))
    1 < dim <= N || throw(ArgumentError("selected dimension to contract is out of bounds"))
    ax = axes(r)
    axC = axes(C)
    (preidx = ax[1:dim-1]) == axC[1:dim-1] || throw(BoundsError("array axes incompatible (check size or indexing)"))
    (sufidx = ax[dim:N_]) == axC[dim+1:N] || throw(BoundsError("array axes incompatible (check size or indexing)"))

    preI = CartesianIndices(preidx)
    sufI = CartesianIndices(sufidx)

    u = k*x                 # unitless, could be complex
    @assert u isa Real "Hermitian series can only be evaluated at real arguments"
    s = size(C, dim);
    iseven(s) && throw(ArgumentError("Hermitian series must have axes with odd lengths"))
    c = firstindex(C, dim)  # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    e = complex(one(u))     # initial offset phase

    # apply global Fourier multiplier z0*(im*k)^a (k^a could change type)
    z = _iszero(a)  ? e            :
        _isone(a)   ? e*im*k       :
        _istwo(a)   ? e*(im*k)^2   : e*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = _iszero(a)  ? z : 0z

    for suffix in sufI, prefix in preI
        @inbounds r[prefix, suffix] = b * C[prefix, c, suffix]
    end

    s == 1 && return r      # fast return for singleton dimensions

    dz = cis(u)   # forward-backward-winding phase steps
    _i = i_ = c
    for n in 1:M    # symmetric winding loop
        z *= dz     # forward--winding phase

        _i += 1     # forward--winding index
        i_ -= 1     # backward-winding index

        # winding coefficients
        b = _iszero(a)  ? z     :
            _isone(a)   ? z*n   :
            _istwo(a)   ? z*n^2 : z*_pow(n, a)

        for suffix in sufI, prefix in preI
            @inbounds r[prefix, suffix] += _mul(b, C[prefix, _i, suffix]) + _mul(conj(b), C[prefix, i_, suffix])
        end
    end
    return r
end

struct HermitianFourierSeries{N,iip,C<:AbstractArray,A<:NTuple{N,Any},T,F} <: AbstractFourierSeries{N,T,iip}
    c::C
    a::A
    t::NTuple{N,T}
    f::NTuple{N,F}
end

function _check_coeffs_hermsym(c::AbstractArray, o::NTuple)
    ax  = axes(c)
    _ax = ax[ndims(c)-length(o)+1:ndims(c)]
    ax_ = ax[1:ndims(c)-length(o)]
    for i in CartesianIndices(ax_)
        _c = @view c[i,CartesianIndices(_ax)]
        # TODO single, not double, check
        all(c -> c[begin] == c[end]', zip(_c, Iterators.reverse(_c))) || return false
    end
    return true
end
function _check_coeffs_centered(c::AbstractArray, o::NTuple)
    @assert ndims(c) >= length(o)
    all(zip(axes(c)[end-length(o)+1:end],o)) do (ax, o)
        len = length(ax)
        isodd(len) && iszero(ax[div(len,2)+begin]+o)
    end
end

#=
Plan for inplace evaluation
If `f` is an inplace evaluator, the Hermitian evaluator may evaluate a different result:
- when `ndims(f)+1 = ndims(f.c)`, the Hermitian evaluator will do the same as
  `f`, which is to apply the same summation to each element, like a batch evaluator
- when `ndims(f)+2 = ndims(f.c)`, and `eltype(f.c) isa Number` the Hermitian
  evaluator will treat the inner dimensions as the indices of an operator. The output
- when `ndims(f)+3 = ndims(f.c)`, and `eltype(f.c) isa Number` the Hermitian
  evaluator will treat the two inner dimensions as the indices of an operator,
  and the third innermost dimension as a batching dimension
- Otherwise, the behavior of the Hermitian evaluators is not defined (it will error)
=#
"""
    HermitianFourierSeries(f::FourierSeries)

Construct an efficient evaluator for an operator-valued [`FourierSeries`](@ref)
whose coefficients have Hermitian symmetry, i.e. ``C_{i} = C_{-i}^{\\dagger}``.
This evaluator guarantees that its output `ishermitian`, and is roughly 2x
faster than a `FourierSeries` and uses half of the memory.

Inplace series are not currently supported
"""
function HermitianFourierSeries(f::FourierSeries)
    isinplace(f) && throw(ArgumentError("inplace series are not supported"))
    _check_coeffs_hermsym(f.c, f.o) || throw(ArgumentError("Fourier coefficients do not have Hermitian symmetry"))
    _check_coeffs_centered(f.c, f.o) || throw(ArgumentError("Fourier coefficients are not centered"))
    ax  = axes(f.c)
    _ax = ax[ndims(f.c)-length(f.o)+2:ndims(f.c)]
    ax_ = ax[1:ndims(f.c)-length(f.o)]
    _ax_ = ax[ndims(f.c)-length(f.o)+1]
    c = f.c[CartesianIndices(ax_),_ax_[begin+div(length(_ax_),2):end],CartesianIndices(_ax)] # copy
    return HermitianFourierSeries{ndims(f),isinplace(f),typeof(c),typeof(f.a),eltype(f.t),eltype(f.f)}(c, f.a, f.t, f.f)
end

function evaluate!(c, s::HermitianFourierSeries{1}, x)
    f = freq2rad(s.f[1])
    if isinplace(s)
        error("not implemented")
        # return hermitian_fourier_contract!(c, s.c, x, f, s.a[1])
    else
        return hermitian_fourier_evaluate(s.c, x, f, s.a[1])
    end
end

function allocate(s::HermitianFourierSeries, x, ::Val{d}) where {d}
    if isinplace(s) || ndims(s) > 1
        M = ndims(s.c)-ndims(s)
        return fourier_allocate(s.c, x, freq2rad(s.f[d]), s.a[d], Val(M+d))
    elseif isinplace(s) && ndims(s) == 1
        error("not implemented")
        # return hermitian_fourier_allocate(s.c, x, freq2rad(s.f[d]), s.a[d])
    else
        return nothing
    end
end

function contract!(c, s::HermitianFourierSeries, x, dim::Val{d}) where {d}
    if d == 1
        error("not implemented")
        # hermitian_fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d])
    else
        M = ndims(s.c)-ndims(s)
        hermitian_fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d], Val(M+d))
    end
    t = deleteat_(s.t, dim)
    f = deleteat_(s.f, dim)
    a = deleteat_(s.a, dim)
    return HermitianFourierSeries{ndims(s)-1,isinplace(s),typeof(c),typeof(a),eltype(t),eltype(f)}(c, a, t, f)
end

period(s::HermitianFourierSeries) = s.t
frequency(s::HermitianFourierSeries) = s.f

function nextderivative(s::HermitianFourierSeries, dim)
    a = raise_multiplier(s.a, dim)
    return HermitianFourierSeries{ndims(s),isinplace(s),typeof(s.c),typeof(a),eltype(s.t),eltype(s.f)}(s.c, a, s.t, s.f)
end
