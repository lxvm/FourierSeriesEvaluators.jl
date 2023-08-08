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
struct FourierSeries{N,C,P,K,A,O,Q} <: AbstractFourierSeries{N}
    c::C
    p::P
    k::K
    a::A
    o::O
    q::Q
    function FourierSeries(c::C, p::P, k::K, a::A, o::O, q::Q) where {N,T,C<:AbstractArray{T,N},P<:Tuple{Vararg{Any,N}},K<:Tuple{Vararg{Any,N}},A<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}},Q<:Tuple{Vararg{Any,N}}}
        return new{N,C,P,K,A,O,Q}(c, p, k, a, o, q)
    end
end

fill_ntuple(e::Union{Number,Val}, N) = ntuple(_ -> e, N)
fill_ntuple(e::Tuple, _) = e
fill_ntuple(e::AbstractArray, _) = tuple(e...)

function FourierSeries(coeffs::AbstractArray; period=2pi, deriv=0, offset=0, shift=nothing)
    N = ndims(coeffs)
    period_ = fill_ntuple(period, N)
    deriv_  = fill_ntuple(deriv,  N)
    offset_ = fill_ntuple(offset, N)
    shift_  = fill_ntuple(shift === nothing ? map(zero, period_) : shift,  N)
    return FourierSeries(coeffs, period_, map(p -> 2pi/p, period_), deriv_, offset_, shift_)
end

function allocate(f::FourierSeries, x, dim::Val{d}) where {d}
    return fourier_allocate(f.c, x, f.k[d], f.a[d], dim)
end

deleteat_(t::Tuple, v::Val{i}) where {i} = deleteat__(v, t...)
deleteat__(::Val{i}, t1, t...) where {i} = (t1, deleteat__(Val(i-1), t...)...)
deleteat__(::Val{1}, t1, t...) = t

function contract!(c, f::FourierSeries, x, dim::Val{d}) where {d}
    fourier_contract!(c, f.c, x-f.q[d], f.k[d], f.a[d], f.o[d], dim)
    p = deleteat_(f.p, dim)
    k = deleteat_(f.k, dim)
    a = deleteat_(f.a, dim)
    o = deleteat_(f.o, dim)
    q = deleteat_(f.q, dim)
    return FourierSeries(c, p, k, a, o, q)
end

evaluate(f::FourierSeries{1}, x::NTuple{1}) =
    fourier_evaluate(f.c, x[1]-f.q[1], f.k[1], f.a[1], f.o[1])

period(f::FourierSeries) = f.p

show_dims(f::FourierSeries) = Base.dims2string(length.(axes(f.c))) * " "
show_details(f::FourierSeries) = " with $(eltype(f.c)) coefficients, $(f.a) derivative, $(f.o) offset, $(f.q) shift"

"""
    ManyFourierSeries(fs::AbstractFourierSeries{N}...; period) where {N}

Represents a tuple of Fourier series of the same dimension and
contracts them all simultaneously.
"""
struct ManyFourierSeries{N,F,P} <: AbstractFourierSeries{N}
    fs::F
    p::P
    function ManyFourierSeries(fs::Tuple{Vararg{AbstractFourierSeries{N}}}, p::Tuple{Vararg{Any,N}}) where {N}
        return new{N,typeof(fs),typeof(p)}(fs, p)
    end
end
function ManyFourierSeries(fs::AbstractFourierSeries{N}...; period=2pi) where {N}
    return ManyFourierSeries(fs, fill_ntuple(period, N))
end

function allocate(fs::ManyFourierSeries, x, dim)
    k = inv(period(fs, dim))
    return map(f -> allocate(f, x*k*period(f, dim), dim), fs.fs)
end

function contract!(cs, fs::ManyFourierSeries, x, dim)
    k = inv(period(fs, dim))
    return ManyFourierSeries(map((c,f) -> contract!(c, f, x*k*period(f, dim), dim), cs, fs))
end

function evaluate(fs::ManyFourierSeries{N}, x::NTuple{N}) where {N}
    k = map(inv, period(fs))
    return map(f -> evaluate(f, map(*, x,k,period(f))), fs.fs)
end

period(f::ManyFourierSeries) = f.p

show_details(fs::ManyFourierSeries) = " with $(length(fs.fs)) series"
