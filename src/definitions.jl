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
    contract!(g::AbstractFourierSeries, f::AbstractFourierSeries, x::Number, dim::Type{Val{d}})

An in-place version of `contract`, however the argument `dim` must be a `Val{d}`
in order to dispatch to the specific contract method. This should return `f`.

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

"""
    coefficients(f::AbstractFourierSeries)

Return the underlying array(s) representing the Fourier series.
"""
function coefficients end

evaluate(f::AbstractFourierSeries{N}, x::NTuple{N}) where N =
    evaluate(contract(f, x[N], Val(N)), x[1:N-1])

# docstring in type definition above
(f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))

Base.ndims(::AbstractFourierSeries{N}) where N = N

"""
    eltype(::AbstractFourierSeries{N,T}) where {N,T}

Returns `T`, the type of the input data to the Fourier series. For the output
type, see [`fourier_type`](@ref)
"""
Base.eltype(::Type{<:AbstractFourierSeries{N,T}}) where {N,T} = T


# helper functions for types

"""
    phase_type(x) = Base.promote_op(cis, eltype(x))

Returns the type of `exp(im*x)`.
"""
phase_type(x) = Base.promote_op(cis, typeof(one(eltype(x))))

deriv_type(x, a) = complex(typeof(inv(oneunit(eltype(x))^a)))

"""
    fourier_type(::Type{T}, x) where T = Base.promote_op(*, T, phase_type(x))
"""
fourier_type(::Type{T}, x) where T =
    Base.promote_op(*, T, phase_type(x))

"""
    fourier_type(C::AbstractFourierSeries, x) = typeof(f(x))

Returns the output type of the Fourier series.
"""
function fourier_type(f::AbstractFourierSeries, x)
    return typeof(f(ntuple(_ -> zero(eltype(x)), Val(ndims(f)))))
end

show_dims(f::AbstractFourierSeries) = show_dims(f, coefficients(f)) * " "
show_dims(f, t::Tuple) = "(" * join(map(s -> show_dims(f,s), t), ", ") * ")"
show_dims(_, A::AbstractArray) = Base.dims2string(length.(axes(A)))
show_dims(f, ::Nothing) = "$(ndims(f))-dimensional "
show_details(::AbstractFourierSeries) = ""

Base.summary(f::AbstractFourierSeries) =
    string(show_dims(f), nameof(typeof(f)), " with $(eltype(f)) coefficients & ", period(f), " periodicity", show_details(f))
Base.show(io::IO, f::AbstractFourierSeries) = print(io, summary(f))
