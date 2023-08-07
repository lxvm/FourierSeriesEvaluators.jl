"""
    AbstractFourierSeries{N}

A supertype for Fourier series that are periodic maps ``\\R^N \\to V`` where
``V`` is any vector space. Typically these can be
represented by `N`-dimensional arrays whose elements belong to the vector space.
No assumptions are made on the input type (it can be real, complex or otherwise) or the
output type (which doesn't have to support vector options)

    (f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Evaluate the Fourier series at the given point (see [`evaluate`](@ref)).
"""
abstract type AbstractFourierSeries{N} end

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
    evaluate(f::AbstractFourierSeries, x)

Evaluate the Fourier series at the point `x`. By default `x` is wrapped into a
tuple and the Fourier series is contracted along the outer dimension.

!!! note "For developers"
    Implementations of the interface only need to define a method specializing
    on the concrete type `T` of `f::T` with signature `evaluate(T, ::NTuple{1})`
    while evaluation of the other dimensions can be delegated to [`contract`](@ref).
"""
function evaluate end

# abstract methods

"""
    contract(f::AbstractFourierSeries{N}, x::Number, ::Val{dim}) where {N,dim}

Return another Fourier series of dimension `N-1` by summing over dimension `dim`
of `f` with the phase factors evaluated at `x`.
"""
contract(f::AbstractFourierSeries, x, dim) = contract!(allocate(f, x, dim), f, x, dim)

evaluate(f::AbstractFourierSeries{N}, x::NTuple{N}) where N =
    evaluate(contract(f, x[N], Val(N)), x[1:N-1])

# docstring in type definition above
(f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Base.ndims(::AbstractFourierSeries{N}) where N = N

show_dims(::AbstractFourierSeries{N}) where {N} = "$N-dimensional "
show_details(::AbstractFourierSeries) = ""

Base.summary(f::AbstractFourierSeries) =
    string(show_dims(f), nameof(typeof(f)), show_details(f))
Base.show(io::IO, f::AbstractFourierSeries) = print(io, summary(f))
