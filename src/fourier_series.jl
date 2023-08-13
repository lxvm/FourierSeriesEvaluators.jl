"""
    FourierSeries(coeffs::AbstractArray, [N]; period, offset=0, deriv=0)

Construct a Fourier series whose coefficients are given by the coefficient array array
`coeffs` whose elements should support addition and scalar multiplication, This object
represents the Fourier series
```math
f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} C_{\\vec{n}} \\left( \\prod_{i} \\left( 2\\pi f_{i} (n_{i} + o_{i}) \\sqrt{-1} \\right)^{a_{i}} \\exp\\left(2\\pi f_{i} x_{i} (n_{i} + o_{i}) \\sqrt{-1} \\right) \\right)
```
Here, the indices ``\\mathcal I`` are the `CartesianIndices` of `coeffs`, ``f_{i} =
1/t_{i}`` is the frequency/inverse period, ``a_{i}`` is the order of derivative, and
``o_{i}`` is the index offset. Also, the keywords, which
can either be a single value applied to all dimensions or a tuple/vector describing each
dimension mean
- `period`: The periodicity of the Fourier series, which must be a real number
- `offset`: An offset in the phase index, which must be integer
- `deriv`: The degree of differentiation, implemented as a Fourier multiplier. Can be any
  number `a` such that `x^a` is well-defined, or a `Val(a)`. `a::Integer` performs best.

If inplace evaluation or evaluation of multiple series is desired, the optional argument `N`
when set fixes the number of variables of the Fourier series, which may be less than or
equal to `ndims(coeffs)`, and the series is evaluated inplace, returning the innermost,
continguous `ndims(coeffs)-N` axes evaluated at the variables corresponding to the remaining
outer axes.
"""
struct FourierSeries{S,N,iip,C,A,T,F} <: AbstractFourierSeries{N,T,iip}
    c::C
    a::A
    t::NTuple{N,T}
    f::NTuple{N,F}
    o::NTuple{N,Int}
    function FourierSeries(c::AbstractArray{<:Any,N}, a::NTuple{N,Any}, t::NTuple{N,T}, f::NTuple{N,F}, o::NTuple{N,Integer}) where {N,T,F}
        return new{0,N,false,typeof(c),typeof(a),T,F}(c, a, t, f, o)
    end
    function FourierSeries{S}(c::AbstractArray{<:Any,M}, a::NTuple{N,Any}, t::NTuple{N,T}, f::NTuple{N,F}, o::NTuple{N,Integer}) where {S,M,N,T,F}
        S == M-N || throw(ArgumentError("number of variables inconsistent with coefficient array"))
        S < 0 && throw(ArgumentError("coefficient array cannot have fewer dimensions than variables"))
        return new{S,N,true,typeof(c),typeof(a),T,F}(c, a, t, f, o)
    end
end

function fill_ntuple(e, N)
    if e isa Tuple || e isa AbstractArray
        @assert length(e) == N
        e isa Tuple && return e # avoids a type instability for heterogenous e
        return NTuple{N}(e)
    else
        return ntuple(_ -> e, N)
    end
end

freq2rad(x) = (x+x)*pi

function FourierSeries(coeffs::AbstractArray; period, offset=0, deriv=0)
    (N = ndims(coeffs)) > 0 || throw(ArgumentError("coefficient array must have at least one dimension"))
    t = fill_ntuple(period, N)
    a = fill_ntuple(deriv,  N)
    o = fill_ntuple(offset, N)
    return FourierSeries(coeffs, a, t, map(inv, t), o)
end
function FourierSeries(coeffs::AbstractArray, N::Integer; period, offset=0, deriv=0)
    N > 0 || throw(ArgumentError("At least one variable is required"))
    t = fill_ntuple(period, N)
    a = fill_ntuple(deriv,  N)
    o = fill_ntuple(offset, N)
    return FourierSeries{ndims(coeffs)-N}(coeffs, a, t, map(inv, t), o)
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
    fourier_contract!(c, s.c, x, freq2rad(s.f[d]), s.a[d], s.o[d], Val(S+d))
    t = deleteat_(s.t, dim)
    f = deleteat_(s.f, dim)
    o = deleteat_(s.o, dim)
    a = deleteat_(s.a, dim)
    return isinplace(s) ? FourierSeries{S}(c, a, t, f, o) : FourierSeries(c, a, t, f, o)
end

function evaluate!(c, s::FourierSeries{S,1}, x) where {S}
    f = freq2rad(s.f[1])
    if isinplace(s)
        return fourier_contract!(c, s.c, x, f, s.a[1], s.o[1], Val(S+1))
    else
        return fourier_evaluate(s.c, (x,), (f,), s.a, s.o)
    end
end

period(s::FourierSeries) = s.t
frequency(s::FourierSeries) = s.f

show_dims(s::FourierSeries) = Base.dims2string(length.(axes(s.c)))
show_details(s::FourierSeries) = " with $(eltype(s.c)) coefficients, $(s.a) derivative, $(s.o) offset"

"""
    ManyFourierSeries(fs::AbstractFourierSeries{N,T,iip}...; period) where {N,T,iip}

Represents a tuple of Fourier series of the same dimension and
contracts them all simultaneously. All the series are required to be either inplace or not.
"""
struct ManyFourierSeries{N,T,iip,S,F} <: AbstractFourierSeries{N,T,iip}
    s::S
    t::NTuple{N,T}
    f::NTuple{N,F}
    function ManyFourierSeries{iip}(s::Tuple{Vararg{AbstractFourierSeries{N,T,iip}}}, t::NTuple{N,T}, f::NTuple{N,F}) where {N,T,F,iip}
        return new{N,T,iip,typeof(s),T}(s, t, f)
    end
end
function ManyFourierSeries(s::AbstractFourierSeries{N,T,iip}...; period) where {N,T,iip}
    t = fill_ntuple(period, N)
    return ManyFourierSeries{iip}(s, t, map(inv, t))
end

function allocate(ms::ManyFourierSeries, x, dim)
    k = frequency(ms, dim)
    return map(s -> allocate(s, x*k*period(s, dim), dim), ms.s)
end

function contract!(cs, ms::ManyFourierSeries, x, dim)
    f = frequency(ms, dim)
    f_ = deleteat_(frequency(ms), dim)
    t_ = deleteat_(period(ms), dim)
    return ManyFourierSeries{isinplace(ms)}(map((c,s) -> contract!(c, s, x*f*period(s, dim), dim), cs, ms.s), t_, f_)
end

function evaluate!(cs, ms::ManyFourierSeries{1}, x)
    f = frequency(ms, 1)
    return map((c,s) -> evaluate!(c, s, x*f*period(s,1)), cs, ms.s)
end

period(ms::ManyFourierSeries) = ms.t
frequency(ms::ManyFourierSeries) = ms.f

show_details(ms::ManyFourierSeries) = " with $(length(ms.s)) series"

# Differentiating Fourier series

raise_multiplier(a) = a + 1
raise_multiplier(::Val{a}) where {a} = Val(a+1)

function raise_multiplier(t, ::Val{d}) where {d}
    return ntuple(n -> n == d ? raise_multiplier(t[n]) : t[n], Val(length(t)))
end

function nextderivative(s::FourierSeries{S}, dim) where {S}
    a = raise_multiplier(s.a, dim)
    if isinplace(s)
        return FourierSeries{S}(s.c, a, s.t, s.f, s.o)
    else
        return FourierSeries(s.c, a, s.t, s.f, s.o)
    end
end

function nextderivative(ms::ManyFourierSeries, dim)
    return ManyFourierSeries{isinplace(ms)}(map(s -> nextderivative(s, dim), ms.s), period(ms), frequency(ms))
end

struct DerivativeSeries{O,N,T,iip,F,DF} <: AbstractFourierSeries{N,T,iip}
    f::F
    df::DF
    function DerivativeSeries{O}(f::AbstractFourierSeries{N,T}, df::ManyFourierSeries{N,T}) where {O,N,T}
        return new{O,N,T,isinplace(f),typeof(f),typeof(df)}(f, df)
    end
end

"""
    DerivativeSeries{O}(f::AbstractFourierSeries)

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

For this routine to work, `f` must implement [`nextderivative`](@ref).
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
    return ManyFourierSeries(df, ds.df.s..., period=period(ds.df))
end
function nextderivative(ds::DerivativeSeries, dim)
    df = nextderivative(nextderivative(ds.f, dim), dim)
    return ManyFourierSeries(df, ds.df.s..., period=period(ds.df))
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

function evaluate!(cache, ds::DerivativeSeries{O,1}, x) where {O}
    fx = evaluate!(cache[1], ds.f, x)
    df = nextderivative(ds, Val(1))
    dfx = evaluate!(cache[2], df, x * period(df, 1) * frequency(ds, 1))
    return O === 1 ? (fx, dfx) : (fx..., dfx)
end

period(ds::DerivativeSeries) = period(ds.f)
frequency(ds::DerivativeSeries) = frequency(ds.f)

show_details(::DerivativeSeries{O}) where {O} = " of order $O"

"""
    JacobianSeries

Alias for a [`DerivativeSeries`](@ref) of order 1
"""
const JacobianSeries = DerivativeSeries{1}

"""
    HessianSeries

Alias for a [`DerivativeSeries`](@ref) of order 2
"""
const HessianSeries  = DerivativeSeries{2}
