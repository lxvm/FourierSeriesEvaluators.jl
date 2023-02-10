"""
    FourierSeries(coeffs::AbstractArray; period=2pi, offset=0, deriv=0, shift=0)

Construct a Fourier series whose coefficients are given by the coefficient array
array `coeffs` whose elements should support addition and scalar multiplication,
This type represents the Fourier series
```math
f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} C_{\\vec{n}} \\exp(i2\\pi\\vec{k}_{\\vec{n}}\\cdot\\overrightarrow{x})
```
The indices ``\\vec{n}`` are the `CartesianIndices` of `coeffs`. Also, the
keywords, which can either be a single value applied to all dimensions or a
collection describing each dimension mean
- `period`: The periodicity of the Fourier series. Equivalent to ``2\\pi/k``
- `offset`: An offset in the phase index, which must be integer
- `deriv`: The degree of differentiation, implemented as a Fourier multiplier
- `shift`: A translation `q` such that the evaluation point `x` is shifted to `x-q`
"""
struct FourierSeries{N,T,C,K,A,O,Q} <: AbstractFourierSeries{N,T}
    c::C
    k::K
    a::A
    o::O
    q::Q
    FourierSeries{N,T}(c::C, k::K, a::A, o::O, q::Q) where {N,T,C<:AbstractArray{T,N},K<:NTuple{N},A<:NTuple{N},O<:NTuple{N},Q<:NTuple{N}} =
        new{N,T,C,K,A,O,Q}(c, k, a, o, q)
end

fill_ntuple(e, N) =
    e isa Union{Tuple{Vararg{Any,N}},AbstractVector} ? promote(e...) : ntuple(_ -> e, N)

function FourierSeries(coeffs::AbstractArray{T,N}; period=2pi, deriv=Val(0), offset=0, shift=0.0) where {T,N}
    period = fill_ntuple(period, N)
    deriv  = fill_ntuple(deriv,  N)
    offset = fill_ntuple(offset, N)
    shift  = fill_ntuple(shift,  N)
    FourierSeries{N,T}(coeffs, 2pi ./ period, deriv, offset, shift)
end

period(f::FourierSeries) = 2pi ./ f.k

deleteat_(t::NTuple{N}, ::Val{i}) where {N,i} = ntuple(n -> t[n+(n>=i)], Val(N-1))

function contract(f::FourierSeries{N,T}, x::Number, ::Val{dim}) where {N,T,dim}
    c = fourier_contract(f.c, x-f.q[N], f.k[dim], f.a[dim], f.o[dim], Val(dim))
    k = deleteat_(f.k, Val(dim))
    a = deleteat_(f.a, Val(dim))
    o = deleteat_(f.o, Val(dim))
    q = deleteat_(f.q, Val(dim))
    FourierSeries{N-1,eltype(c)}(c, k, a, o, q)
end

evaluate(f::FourierSeries{1}, x::NTuple{1}) =
    fourier_evaluate(f.c, x[1]-f.q[1], f.k[1], f.a[1], f.o[1])


"""
    InplaceFourierSeries(coeffs::AbstractArray; period=2pi, offset=0, deriv=0, shift=0)

Similar to [`FourierSeries`](@ref) except that it doesn't allocate new arrays
for every call to `contract` and `contract` is limited to the outermost
dimension/variable of the series.
"""
struct InplaceFourierSeries{N,T,F,C,K,A,O,Q} <: AbstractInplaceFourierSeries{N,T}
    f::F
    c::C
    k::K
    a::A
    o::O
    q::Q
    InplaceFourierSeries{N,T}(f::F, c::C, k::K, a::A, o::O, q::Q) where {N,T,F,C<:AbstractArray{T,N},K,A,O,Q} =
        new{N,T,F,C,K,A,O,Q}(f, c, k, a, o, q)
end

function InplaceFourierSeries(coeffs::AbstractArray{T,N}; period=2pi, deriv=Val(0), offset=0, shift=0.0) where {T,N}
    period = fill_ntuple(period, N)
    deriv  = fill_ntuple(deriv,  N)
    offset = fill_ntuple(offset, N)
    shift  = fill_ntuple(shift,  N)
    v = view(coeffs, ntuple(n -> n==N ? first(axes(coeffs,n)) : axes(coeffs,n), Val{N}())...)
    c = similar(v, fourier_type(T,eltype(period)))
    f = InplaceFourierSeries(c; period=period[1:N-1], deriv=deriv[1:N-1], offset=offset[1:N-1], shift=shift[1:N-1])
    InplaceFourierSeries{N,T}(f, coeffs, 2pi/period[N], deriv[N], offset[N], shift[N])
end
function InplaceFourierSeries(coeffs::AbstractArray{T,0}; period=(), deriv=(), offset=(), shift=()) where T
    InplaceFourierSeries{0,T}((), coeffs, period, deriv, offset, shift)
end

period(f::InplaceFourierSeries{0}) = f.k
period(f::InplaceFourierSeries) = (period(f.f)..., 2pi/f.k)

function contract!(f::F, x::Number, ::Val{N}) where {N,F<:InplaceFourierSeries{N}}
    fourier_contract!(f.f.c, f.c, x-f.q, f.k, f.a, f.o, Val(N))
    return f.f
end

evaluate(f::InplaceFourierSeries{1}, x::NTuple{1}) =
    fourier_evaluate(f.c, x[1]-f.q, f.k, f.a, f.o)


"""
    ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}

Represents a tuple of Fourier series of the same dimension and periodicity and
contracts them all simultaneously.
"""
struct ManyFourierSeries{N,T,F} <: AbstractFourierSeries{N,T}
    fs::F
    ManyFourierSeries{N,T}(fs::F) where {N,T,F} = new{N,T,F}(fs)
end
ManyFourierSeries(fs::AbstractFourierSeries{N}...) where N =
    ManyFourierSeries{N,Tuple{map(eltype, fs)...}}(fs)

function period(fs::ManyFourierSeries)
    ref = period(fs.fs[1])
    @assert all(map(==(ref), map(period, Base.tail(fs.fs)))) "all periods should match"
    ref
end

function contract(fs::ManyFourierSeries{N,T}, x::Number, ::Val{dim}) where {N,T,dim}
    fxs = map(f -> contract(f, x, Val(dim)), fs.fs)
    ManyFourierSeries{N-1,Tuple{map(eltype, fxs)...}}(fxs)
end

evaluate(fs::ManyFourierSeries{N}, x::NTuple{N}) where N =
    map(f -> evaluate(f, x), fs.fs)

