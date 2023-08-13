"""
    AbstractFourierSeries{N,T,iip}

A supertype for multidimensional Fourier series objects. Given a `f::AbstractFourierSeries`,
you can evaluate it at a point `x` with `f(x)`, where `x` is a vector (or scalar if `f` is
1d).

Fourier series are periodic maps ``T^N \\to V`` where `T` is a space of real numbers and
``V`` is any vector space. Typically, a Fourier series can be represented by `N`-dimensional
arrays whose elements belong to the vector space. If `iip` is `true`, then ``V`` is assumed
to have mutable elements and inplace array operations are used. Otherwise, ``V`` is assumed
to be immutable. The period of the series should be specified by values of type `T`,
although no restriction is placed on the inputs to the series, e.g. arguments of type
`Complex{T}` are OK. Additionally, if the caller wants to determine the floating-point
precision of the Fourier coefficients, `T` and the arguments must both have that precision.
"""
abstract type AbstractFourierSeries{N,T,iip} end

# interface: subtypes of AbstractFourierSeries must implement the following

"""
    allocate(f::AbstractFourierSeries{N}, x, ::Val{d}) where {N,d}

Return a cache that can be used by [`contract!`](@ref) to store the result of
contracting the coefficients of `f` along axis `d` using an input `x`.
"""
function allocate end

"""
    contract!(cache, f::AbstractFourierSeries{N}, x, ::Val{d}) where {N,d}

Return another Fourier series of dimension `N-1` by summing over dimension `d` of `f` with
the phase factors evaluated at `x` and using the storage in `cache` created by a call to
[`allocate`](@ref)
"""
function contract! end

"""
    evaluate!(cache, f::AbstractFourierSeries{1}, x)

Evaluate the Fourier series at the point `x` using a `cache` for inplace evaluation created
by a call to [`allocate`](@ref). If the series is inplace, the `cache` storage may be used
as the return value, and if the series is not inplace the cache may be unused.
"""
function evaluate! end

"""
    period(f::AbstractFourierSeries, [dim])

Return a tuple containing the periodicity of `f`. Optionally you can specify a dimension to
just get the period of that dimension. This should have the floating-point precision of the
input used for the Fourier series evaluation.
"""
function period end

"""
    frequency(f::AbstractFourierSeries, [dim]) == map(inv, period(f, [dim]))

Return a tuple containing the frequency, or inverse of the period, of `f`. Optionally you
can specify a dimension to just get the frequency of that dimension.
"""
function frequency end

"""
    nextderivative(f::AbstractFourierSeries, ::Val{d}) where {d}

This method returns a new series that evaluates the derivative of `f` with respect to its
`d`th variable. This method is optional for normal evaluation, but [`DerivativeSeries`](@ref)
requires it.
"""
function nextderivative end

# abstract methods

isinplace(::AbstractFourierSeries{N,T,iip}) where {N,T,iip} = iip

function contract(f::AbstractFourierSeries, x, dim)
    return contract!(allocate(f, x, dim), f, x, dim)
end

function evaluate(f::AbstractFourierSeries{1}, (x,)::NTuple{1})
    return evaluate!(allocate(f, x, Val(1)), f, x)
end
function evaluate(f::AbstractFourierSeries{N}, x::NTuple{N,Any}) where {N}
    return evaluate(contract(f, x[N], Val(N)), x[1:N-1])
end

for name in (:period, :frequency)
    @eval $name(f::AbstractFourierSeries, dim::Integer) = $name(f)[dim]
    @eval $name(f::AbstractFourierSeries, ::Val{d}) where {d} = $name(f)[d]
end

# docstring in type definition above
function (f::AbstractFourierSeries)(x)
    (N = ndims(f)) == length(x) || throw(ArgumentError("number of input variables doesn't match those in series"))
    return evaluate(f, NTuple{N}(x))
end
Base.ndims(::AbstractFourierSeries{N}) where N = N

show_dims(::AbstractFourierSeries{N}) where {N} = "$N-dimensional"
show_inplace(f::AbstractFourierSeries) = isinplace(f) ? ", inplace, " : " "
show_period(f::AbstractFourierSeries) = "and $(period(f))-periodic "
show_details(::AbstractFourierSeries) = ""

Base.summary(f::AbstractFourierSeries) =
    string(show_dims(f), show_inplace(f), show_period(f), nameof(typeof(f)), show_details(f))
Base.show(io::IO, f::AbstractFourierSeries) = print(io, summary(f))
