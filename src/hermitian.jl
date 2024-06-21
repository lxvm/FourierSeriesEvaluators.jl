# here we make use of the fact that for coefficients that satisfy A[i] = A[-i]' produce
# hermitian-valued Fourier series since z*A[i] + z'*A[-i] = (z*A[i]) + (z*A[i])'

_hermitianpart(a::Complex) = real(a)
_hermitianpart(A::AbstractMatrix) = Hermitian(A)
@inline @fastmath _add(x, y) = x + y
@fastmath function hermitian_fourier_evaluate(C::AbstractVector{T}, x, k=inv(oneunit(x)), a=Val(0)) where {T}
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
    @inbounds r = _hermitianpart(b * C[i]) # unroll first loop iteration
    s == 1 && return r      # fast return for singleton dimensions

    dz = cis(1u)  # forward-backward-winding phase steps
    for n in 1:s-1          # symmetric winding loop
        z *= dz             # forward--winding phase
        i += 1              # forward--winding index
        # winding coefficients
        b = _iszero(a)  ? z     :
            _isone(a)   ? z*n   :
            _istwo(a)   ? z*n^2 : z*_pow(n, a)

        @inbounds B = b * C[i]
        r += _add(_hermitianpart(B), _hermitianpart(B')) # _add function barrier allows control of fastmath
    end
    return r
end

struct HermitianFourierSeries{N,iip,C<:AbstractArray,A<:NTuple{N,Any},T,F} <: AbstractFourierSeries{N,T,iip}
    c::C
    a::A
    t::NTuple{N,T}
    f::NTuple{N,F}
end

function HermitianFourierSeries(coeffs::AbstractArray; period, deriv=0)
    (N = ndims(coeffs)) > 0 || throw(ArgumentError("coefficient array must have at least one dimension"))
    t = fill_ntuple(period, N)
    a = fill_ntuple(deriv,  N)
    f = map(inv, t)
    return HermitianFourierSeries{N,false,typeof(coeffs),typeof(a),eltype(t),eltype(f)}(coeffs, a, t, f)
end
function HermitianFourierSeries(coeffs::AbstractArray, N::Integer; period, deriv=0)
    error("not implemented") # have to decide how the array is entered
    N > 0 || throw(ArgumentError("At least one variable is required"))
    ndims(coeffs)-N < 0 && throw("coefficient array cannot have fewer dimensions than variables")
    t = fill_ntuple(period, N)
    a = fill_ntuple(deriv,  N)
    f = map(inv, t)
    return HermitianFourierSeries{N,true,typeof(coeffs),typeof(a),eltype(t),eltype(f)}(coeffs, a, t, f)
end

function evaluate!(c, s::HermitianFourierSeries{1}, x)
    f = freq2rad(s.f[1])
    if isinplace(s)
        error("not implemented")
        # return hermitian_fourier_contract!(c, s.c, x, f, s.a[1], Val(ndims(s.c)))
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
    else
        return nothing
    end
end

function contract!(c, s::HermitianFourierSeries, x, dim::Val{d}) where {d}
    M = ndims(s.c)-ndims(s)
    d == 1 && error("not implemented")
    fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d], -div(size(s.c, d),2)-1, Val(M+d))
    t = deleteat_(s.t, dim)
    f = deleteat_(s.f, dim)
    a = deleteat_(s.a, dim)
    return HermitianFourierSeries{ndims(s)-1,isinplace(s),typeof(c),typeof(a),eltype(t),eltype(f)}(c, a, t, f)
end

period(s::HermitianFourierSeries) = s.t
frequency(s::HermitianFourierSeries) = s.f
