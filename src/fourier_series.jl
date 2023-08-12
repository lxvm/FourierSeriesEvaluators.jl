"""
    FourierSeries(coeffs::AbstractArray, [N]; period=2pi, offset=0, deriv=0, shift=0)

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
If the optional argument `N` is set, it fixes the number of variables of the Fourier series,
which may be less than or equal to `ndims(coeffs)`, and the series is evaluated inplace.
"""
struct FourierSeries{S,N,iip,C,P,F,A,O,Q} <: AbstractFourierSeries{N,iip}
    c::C
    p::P
    f::F
    a::A
    o::O
    q::Q
    function FourierSeries(c::C, p::P, f::F, a::A, o::O, q::Q) where {N,T,C<:AbstractArray{T,N},P<:NTuple{N,Any},F<:NTuple{N,Any},A<:NTuple{N,Any},O<:NTuple{N,Any},Q<:NTuple{N,Any}}
        return new{0,N,false,C,P,F,A,O,Q}(c, p, f, a, o, q)
    end
    function FourierSeries{S}(c::C, p::P, f::F, a::A, o::O, q::Q) where {S,M,N,T,C<:AbstractArray{T,M},P<:NTuple{N,Any},F<:NTuple{N,Any},A<:NTuple{N,Any},O<:NTuple{N,Any},Q<:NTuple{N,Any}}
        S == M-N || throw(ArgumentError("number of variables inconsistent with coefficient array"))
        S < 0 && throw(ArgumentError("coefficient array cannot have fewer dimensions than variables"))
        return new{S,N,true,C,P,F,A,O,Q}(c, p, f, a, o, q)
    end
end

fill_ntuple(e::Union{Number,Val}, N) = ntuple(_ -> e, N)
fill_ntuple(e::Tuple, _) = e
fill_ntuple(e::AbstractArray, _) = tuple(e...)

freq2rad(x) = (x+x)*pi
default_period(::Type{T}) where {T<:Number} = freq2rad(float(real(one(T))))

function FourierSeries(coeffs::AbstractArray; period=nothing, deriv=0, offset=0, shift=nothing)
    N = ndims(coeffs)
    p = fill_ntuple(period === nothing ? default_period(eltype(eltype(coeffs))) : period, N)
    a = fill_ntuple(deriv,  N)
    o = fill_ntuple(offset, N)
    q = fill_ntuple(shift === nothing ? map(zero, p) : shift,  N)
    return FourierSeries(coeffs, p, map(inv, p), a, o, q)
end
function FourierSeries(coeffs::AbstractArray, N::Integer; period=nothing, deriv=0, offset=0, shift=nothing)
    p = fill_ntuple(period === nothing ? default_period(eltype(eltype(coeffs))) : period, N)
    a = fill_ntuple(deriv,  N)
    o = fill_ntuple(offset, N)
    q = fill_ntuple(shift === nothing ? map(zero, p) : shift,  N)
    return FourierSeries{ndims(coeffs)-N}(coeffs, p, map(inv, p), a, o, q)
end

function allocate(s::FourierSeries{S}, x, ::Val{d}) where {S,d}
    if isinplace(s) || ndims(s) > 1
        return fourier_allocate(s.c, x, freq2rad(s.f[d]), s.a[d], Val(S+d))
    else
        return nothing
    end
end

deleteat_(t::Tuple, v::Val{i}) where {i} = deleteat__(v, t...)
deleteat__(::Val{i}, t1, t...) where {i} = (t1, deleteat__(Val(i-1), t...)...)
deleteat__(::Val{1}, t1, t...) = t

function contract!(c, s::FourierSeries{S}, x, dim::Val{d}) where {S,d}
    fourier_contract!(c, s.c, x-s.q[d], freq2rad(s.f[d]), s.a[d], s.o[d], Val(S+d))
    p = deleteat_(s.p, dim)
    f = deleteat_(s.f, dim)
    a = deleteat_(s.a, dim)
    o = deleteat_(s.o, dim)
    q = deleteat_(s.q, dim)
    if isinplace(s)
        return FourierSeries{S}(c, p, f, a, o, q)
    else
        return FourierSeries(c, p, f, a, o, q)
    end
end

function evaluate(s::FourierSeries{0,1,false}, x::NTuple{1})
    return fourier_evaluate(s.c, map(-, x, s.q), map(freq2rad, s.f), s.a, s.o)
end
evaluate(s::FourierSeries{S,0,true}, ::Tuple{}) where {S} = s.c


period(s::FourierSeries) = s.p
frequency(s::FourierSeries) = s.f

show_dims(s::FourierSeries) = Base.dims2string(length.(axes(s.c)))
show_details(s::FourierSeries) = " with $(eltype(s.c)) coefficients, $(s.a) derivative, $(s.o) offset, $(s.q) shift"

"""
    ManyFourierSeries(fs::AbstractFourierSeries{N,iip}...; period) where {N,iip}

Represents a tuple of Fourier series of the same dimension and
contracts them all simultaneously. All the series are required to be either inplace or not.
"""
struct ManyFourierSeries{N,iip,F,P,K} <: AbstractFourierSeries{N,iip}
    fs::F
    p::P
    k::K
    function ManyFourierSeries{iip}(fs::Tuple{Vararg{AbstractFourierSeries{N,iip}}}, p::NTuple{N,Any}, k::NTuple{N,Any}) where {N,iip}
        return new{N,iip,typeof(fs),typeof(p),typeof(k)}(fs, p, k)
    end
end
function ManyFourierSeries(fs::AbstractFourierSeries{N,iip}...; period=2pi) where {N,iip}
    p = fill_ntuple(period, N)
    return ManyFourierSeries{iip}(fs, p, map(inv, p))
end

function allocate(ms::ManyFourierSeries, x, dim)
    k = frequency(ms, dim)
    return map(s -> allocate(s, x*k*period(s, dim), dim), ms.fs)
end

function contract!(cs, ms::ManyFourierSeries, x, dim)
    k = frequency(ms, dim)
    k_ = deleteat_(frequency(ms), dim)
    p_ = deleteat_(period(ms), dim)
    return ManyFourierSeries{isinplace(ms)}(map((c,s) -> contract!(c, s, x*k*period(s, dim), dim), cs, ms.fs), p_, k_)
end

function evaluate(ms::ManyFourierSeries{N}, x::NTuple{N}) where {N}
    k = frequency(ms)
    return map(s -> evaluate(s, map(*, x,k,period(s))), ms.fs)
end

period(ms::ManyFourierSeries) = ms.p
frequency(ms::ManyFourierSeries) = ms.k

show_details(ms::ManyFourierSeries) = " with $(length(ms.fs)) series"

# Differentiating Fourier series

raise_multiplier(a) = a + 1
raise_multiplier(::Val{a}) where {a} = Val(a+1)

function raise_multiplier(t, ::Val{d}) where {d}
    return ntuple(n -> n == d ? raise_multiplier(t[n]) : t[n], Val(length(t)))
end

function nextderivative(s::FourierSeries{S}, dim) where {S}
    if isinplace(s)
        return FourierSeries{S}(s.c, s.p, s.f, raise_multiplier(s.a, dim), s.o, s.q)
    else
        return FourierSeries(s.c, s.p, s.f, raise_multiplier(s.a, dim), s.o, s.q)
    end
end

function nextderivative(ms::ManyFourierSeries, dim)
    return ManyFourierSeries{isinplace(ms)}(map(s -> nextderivative(s, dim), ms.fs), period(ms), frequency(ms))
end

struct DerivativeSeries{O,N,iip,F,DF} <: AbstractFourierSeries{N,iip}
    f::F
    df::DF
    function DerivativeSeries{O}(f::AbstractFourierSeries{N}, df::ManyFourierSeries{N}) where {O,N}
        return new{O,N,isinplace(f),typeof(f),typeof(df)}(f, df)
    end
end

"""
    DerivativeSeries{O}(f::FourierSeries)

Construct an evaluator of a Fourier series and all of its derivatives up to order `O`, which
must be a positive integer. `O=1` gives the gradient, `O=2` gives the Hessian, and so on.
The derivatives are returned in order as a tuple `(f(x), df(x), d2f(x), ..., dOf(x))` where
the entry of order `O` is given by:
- `O=0`: `f`
- `O=1`: `(dfdx1, ..., dfdxN)`
- `O=2`: `((d2fdx1dx1, ..., d2fdx1dxN), ..., (d2fdxNdxN,))`
- `O=3`: `(((d3fdx1dx1dx1, ..., d3fdx1dx1dxN), ..., (d3fdx1dxNdxN,)), ..., ((d3fdxNdxNdxN,),))`
and so on. The fewest number of contractions are made to compute all derivatives.
As can be seen from the pattern above, the `O`-th derivative with
partial derivatives `i = [a_1 ≤ ... ≤ a_N]` is stored in `ds(x)[O+1][i[1]][i[2]]...[i[N]]`.
These indices are given by the simplical generalization of [triangular
numbers](https://en.wikipedia.org/wiki/Triangular_number). For examples of how to index into
the solution see the unit tests.
"""
function DerivativeSeries{O}(f::AbstractFourierSeries) where {O}
    O isa Integer || throw(ArgumentError("Derivative order must be an integer"))
    if O == 0
        return f
    elseif O > 0
        return DerivativeSeries{O}(DerivativeSeries{O-1}(f), ManyFourierSeries{isinplace(f)}((),period(f),frequency(f)))
    else
        throw(ArgumentError("Derivatives of negative order not supported"))
    end
end

function nextderivative(ds::DerivativeSeries{1}, dim)
    df = nextderivative(ds.f, dim)
    return ManyFourierSeries(df, ds.df.fs..., period=period(ds.df))
end
function nextderivative(ds::DerivativeSeries, dim)
    df = nextderivative(nextderivative(ds.f, dim), dim)
    return ManyFourierSeries(df, ds.df.fs..., period=period(ds.df))
end

function allocate(ds::DerivativeSeries, x, dim)
    f_cache = allocate(ds.f, x, dim)
    df = nextderivative(ds, dim)
    df_cache = allocate(df, x * period(df, dim) * frequency(ds, dim), dim)
    return (f_cache, df_cache)
end

function contract!(cache, ds::DerivativeSeries{O}, x, dim) where {O}
    fx = contract!(cache[1], ds.f, x, dim)
    df = nextderivative(ds, dim)
    dfx = contract!(cache[2], df, x * period(df, dim) * frequency(ds, dim), dim)
    return DerivativeSeries{O}(fx, dfx)
end

function evaluate(ds::DerivativeSeries{O,1}, x::NTuple{1}) where {O}
    fx = evaluate(ds.f, x)
    df = nextderivative(ds, Val(1))
    dfx = evaluate(df, map(*, x, period(df), frequency(ds)))
    return O === 1 ? (fx, dfx) : (fx..., dfx)
end

period(ds::DerivativeSeries) = period(ds.f)
frequency(ds::DerivativeSeries) = frequency(ds.f)

show_details(::DerivativeSeries{O}) where {O} = " of order $O"

const JacobianSeries = DerivativeSeries{1}
const HessianSeries  = DerivativeSeries{2}
