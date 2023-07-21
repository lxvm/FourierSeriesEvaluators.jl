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
    FourierSeries{N,T}(c::C, k::K, a::A, o::O, q::Q) where {N,T,C<:AbstractArray{T,N},K<:Tuple{Vararg{Any,N}},A<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}},Q<:Tuple{Vararg{Any,N}}} =
        new{N,T,C,K,A,O,Q}(c, k, a, o, q)
end

fill_ntuple(e::Union{Number,Val}, N) = ntuple(_ -> e, N)
fill_ntuple(e::Tuple, _) = e
fill_ntuple(e::AbstractArray, _) = tuple(e...)

function FourierSeries(coeffs::AbstractArray{T,N}; period=2pi, deriv=0, offset=0, shift=zero(period)) where {T,N}
    period = fill_ntuple(period, N)
    deriv  = fill_ntuple(deriv,  N)
    offset = fill_ntuple(offset, N)
    shift  = fill_ntuple(shift,  N)
    FourierSeries{N,T}(coeffs, 2pi ./ period, deriv, offset, shift)
end

period(f::FourierSeries) = 2pi ./ f.k

deriv(f::FourierSeries) = f.a

offset(f::FourierSeries) = f.o

shift(f::FourierSeries) = f.q

deleteat_(t::Tuple, v::Val{i}) where {i} = deleteat__(v, t...)
deleteat__(::Val{i}, t1, t...) where {i} = (t1, deleteat__(Val(i-1), t...)...)
deleteat__(::Val{1}, t1, t...) = t

function contract!(c::AbstractArray{T,M}, f::FourierSeries{N,S}, x::Number, ::Val{dim}) where {M,T,N,S,dim}
    fourier_contract!(c, f.c, x-f.q[dim], f.k[dim], f.a[dim], f.o[dim], Val(dim))
    k = deleteat_(f.k, Val(dim))
    a = deleteat_(f.a, Val(dim))
    o = deleteat_(f.o, Val(dim))
    q = deleteat_(f.q, Val(dim))
    return FourierSeries{M,T}(c, k, a, o, q)
end

function contract(f::FourierSeries{N,T}, x::Number, ::Val{dim}) where {N,T,dim}
    r = fourier_allocate(f.c, x, f.k[dim], f.a[dim], Val(dim))
    return contract!(r, f, x, Val(dim))
end

evaluate(f::FourierSeries{1}, x::NTuple{1}) =
    fourier_evaluate(f.c, x[1]-f.q[1], f.k[1], f.a[1], f.o[1])


coefficients(f::FourierSeries) = f.c

show_details(f::FourierSeries) =
    " & $(f.a) derivative & $(f.o) offset & $(f.q) shift"

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

deriv(f::ManyFourierSeries) = map(deriv, f.fs)

offset(f::ManyFourierSeries) = map(offset, f.fs)

shift(f::ManyFourierSeries) = map(shift, f.fs)


function contract(fs::ManyFourierSeries{N}, x::Number, ::Val{dim}) where {N,dim}
    fxs = map(f -> contract(f, x, Val(dim)), fs.fs)
    return ManyFourierSeries{N-1,Tuple{map(eltype, fxs)...}}(fxs)
end

function contract!(cs, fs::ManyFourierSeries{N}, x, ::Val{dim}) where {N,dim}
    gs = map((c,f) -> contract!(c, f, x, Val(dim)), cs, fs)
    return ManyFourierSeries{N-1,Tuple{map(eltype, fs)...}}(gs)
end

evaluate(fs::ManyFourierSeries{N}, x::NTuple{N}) where N =
    map(f -> evaluate(f, x), fs.fs)

show_details(fs::ManyFourierSeries) =
    " & $(length(fs.fs)) element$(length(fs.fs) > 1 ? "s" : "")"

coefficients(f::ManyFourierSeries) = map(coefficients, f.fs)
