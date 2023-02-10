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
    c = fourier_contract(f.c, x, f.k[dim], f.a[dim], f.o[dim], Val(dim))
    k = deleteat_(f.k, Val(dim))
    a = deleteat_(f.a, Val(dim))
    o = deleteat_(f.o, Val(dim))
    q = deleteat_(f.q, Val(dim))
    FourierSeries{N-1,T}(c, k, a, o, q)
end

evaluate(f::FourierSeries{1}, x::NTuple{1}) =
    fourier_evaluate(f.c, x[1]-f.q[1], f.k[1], f.a[1], f.o[1])
evaluate(f::FourierSeries{N}, x::NTuple{N}) where N =
    evaluate(contract(f, x[N]-f.q[N], Val(N)), x[1:N-1])
evaluate(f::FourierSeries, x) =
    evaluate(f, Tuple(x))


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

contract(fs::ManyFourierSeries{N,T}, x::Number, ::Val{dim}) where {N,T,dim} =
    ManyFourierSeries{N-1,T}(map(f -> contract(f, x, Val(dim)), fs.fs))

evaluate(fs::ManyFourierSeries{1}, x::NTuple{1}) =
    map(f -> evaluate(f, x[1]), fs.fs)
evaluate(fs::ManyFourierSeries{N}, x::NTuple{N}) where N =
    evaluate(contract(fs, x[N], Val(N)), x[1:N-1])
evaluate(fs::ManyFourierSeries, x) =
    evaluate(fs, Tuple(x))