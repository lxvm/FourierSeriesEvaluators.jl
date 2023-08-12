"""
    AbstractFourierSeries{N,iip}

A supertype for Fourier series that are periodic maps ``\\R^N \\to V`` where
``V`` is any vector space. If `iip` is `true`, then the series is evaluated inplace using
mutating array operations. Otherwise, the series. Typically these can be
represented by `N`-dimensional arrays whose elements belong to the vector space.
No assumptions are made on the input type (it can be real, complex or otherwise) or the
output type (which doesn't have to support vector options)

    (f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Evaluate the Fourier series at the given point (see [`evaluate`](@ref)).
"""
abstract type AbstractFourierSeries{N,iip} end

# interface: subtypes of AbstractFourierSeries must implement the following

"""
    allocate(f::AbstractFourierSeries, x, dim::Val{d}) where {d}

Return an cache to of dimension one less than `f` so that contracting `f`
along axis `d` at a point `x` can be saved to the cache
"""
function allocate end

"""
    contract!(cache, f::AbstractFourierSeries, x, dim::Val{d}) where {d}

An in-place version of `contract`, however the argument `dim` must be a `Val{d}`
in order to dispatch to the specific contract method. This should return a new
`AbstractFourierSeries` of dimension one less than `f`.
"""
function contract! end

"""
    evaluate(f::AbstractFourierSeries{N}, x::NTuple{N})

Evaluate the Fourier series at the point `x`.

!!! note "For developers"
    Implementations of the interface only need to define a method specializing
    on the concrete type `T` of `f::T` with signature `evaluate(T, ::NTuple{1})`
    while evaluation of the other dimensions can be delegated to [`contract`](@ref).
"""
function evaluate end

"""
    period(f::AbstractFourierSeries, [dim])

Return a tuple containing the periodicity of `f`. Optionally you can specify a dimension to
just get the period of that dimension.
"""
function period end

"""
    frequency(f::AbstractFourierSeries, [dim]) = map(inv, period(f, [dim]))

Return a tuple containing the frequency, or inverse of the period, of `f`. Optionally you
can specify a dimension to just get the frequency of that dimension.
"""
function frequency end

# abstract methods

isinplace(::AbstractFourierSeries{N,iip}) where {N,iip} = iip

"""
    contract(f::AbstractFourierSeries{N}, x::Number, ::Val{dim}) where {N,dim}

Return another Fourier series of dimension `N-1` by summing over dimension `dim`
of `f` with the phase factors evaluated at `x`.
"""
contract(f::AbstractFourierSeries, x, dim) = contract!(allocate(f, x, dim), f, x, dim)

function evaluate(f::AbstractFourierSeries{N}, x::NTuple{N}) where {N}
    return evaluate(contract(f, x[N], Val(N)), x[1:N-1])
end

for name in (:period, :frequency)
    @eval $name(f::AbstractFourierSeries, dim::Integer) = $name(f)[dim]
    @eval $name(f::AbstractFourierSeries, ::Val{d}) where {d} = $name(f)[d]
end

# docstring in type definition above
(f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Base.ndims(::AbstractFourierSeries{N}) where N = N

show_dims(::AbstractFourierSeries{N}) where {N} = "$N-dimensional"
show_inplace(f::AbstractFourierSeries) = isinplace(f) ? ", inplace, " : " "
show_period(f::AbstractFourierSeries) = "and $(period(f))-periodic "
show_details(::AbstractFourierSeries) = ""

Base.summary(f::AbstractFourierSeries) =
    string(show_dims(f), show_inplace(f), show_period(f), nameof(typeof(f)), show_details(f))
Base.show(io::IO, f::AbstractFourierSeries) = print(io, summary(f))
