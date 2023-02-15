"""
    AbstractFourierSeries{N,T}

A supertype for Fourier series that are periodic maps ``\\R^N \\to V`` where
``V`` is any vector space with elements of type `T`. Typically these can be
represented by `N`-dimensional arrays whose elements belong to the vector space.

    (f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Evaluate the Fourier series at the given point (see [`evaluate`](@ref)).
"""
abstract type AbstractFourierSeries{N,T} end

"""
    contract(f::AbstractFourierSeries{N}, x::Number, ::Val{dim}) where {N,dim}

Return another Fourier series of dimension `N-1` by summing over dimension `dim`
of `f` with the phase factors evaluated at `x`.

!!! note "For developers"
    Implementations of the interface need to provide an implementation of
    contract with the same signature as the above that specialize on the
    concrete type of `f`. While any value of `dim` from `1` to `N` is can be
    implemented, it is most important to implement `dim==N`.
"""
function contract end

"""
    AbstractInplaceFourierSeries{N,T} <: AbstractFourierSeries{N,T}

A supertype for Fourier series evaluated in place. These define the `contract!`
method instead of `contract`.
"""
abstract type AbstractInplaceFourierSeries{N,T} <: AbstractFourierSeries{N,T} end

"""
    contract!(f::AbstractInplaceFourierSeries, x::Number, dim::Type{Val{d}})

An in-place version of `contract`, however the argument `dim` must be a `Val{d}`
in order to dispatch to the specific contract method. This should return `f`.

!!! note "For developers"
    An [`AbstractInplaceFourierSeries`](@ref) only needs to implement
    `contract!`, which is set up to be called by [`contract`](@ref)
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

"""
    period(f::AbstractFourierSeries{N}) where {N}

Return a `NTuple{N}` whose `m`-th element corresponds to the period of `f`
along its `m`-th input dimension. Typically, these values set the units of
length for the problem.
"""
function period end


# abstract methods

contract(f::AbstractInplaceFourierSeries{N}, x::Number, ::Val{d}=Val(N)) where {N,d} =
    contract!(f, x, Val(d))

evaluate(f::AbstractFourierSeries{N}, x::NTuple{N}) where N =
    evaluate(contract(f, x[N], Val(N)), x[1:N-1])

# docstring in type definition above
(f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Base.ndims(::AbstractFourierSeries{N}) where N = N

"""
    eltype(::AbstracFourierSeries{N,T}) where {N,T}

Returns `T`, the type of the input data to the Fourier series. For the output
type, see [`fourier_type`](@ref)
"""
Base.eltype(::Type{AbstractFourierSeries{N,T}}) where {N,T} = T


# helper functions for types

"""
    phase_type(x) = Base.promote_op(cis, eltype(x))

Returns the type of `exp(im*x)`.
"""
phase_type(x) = Base.promote_op(cis, eltype(x))

"""
    fourier_type(::Type{T}, x) where T = Base.promote_op(*, T, phase_type(x))
    fourier_type(C::AbstractFourierSeries, x) = fourier_type(eltype(f), x)

Returns the output type of the Fourier series.
"""
fourier_type(::Type{T}, x) where T =
    Base.promote_op(*, T, phase_type(x))
fourier_type(f::AbstractFourierSeries, x) =
    fourier_type(eltype(f), x)
