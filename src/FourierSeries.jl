# Possible TODO for FourierSeries
# - enable reduction over multiple dims simulatneously (?) may not be used
# - replace contract kernels with the fourier_kernel
# - add an argument n to fourier_kernel to set `z = cispi(2n*ξ*x)`

"""
    FourierSeries(coeffs::AbstractArray{T,N}, period::Real)
    FourierSeries(coeffs::AbstractArray{T,N}, period::NTuple{N,<:Real}) where {T,N}
    FourierSeries{N}(coeffs, period::NTuple{N,<:Real}) where {N}

Construct a Fourier series whose coefficients are given by the coefficient array
array `coeffs` whose `eltype` should support addition and scalar multiplication,
and whose periodicity on the `i`th axis is given by `period[i]`. This type
represents the Fourier series
```math
f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} C_{\\vec{n}} \\exp(i2\\pi\\vec{k}_{\\vec{n}}\\cdot\\overrightarrow{x})
```
where ``i = \\sqrt{-1}`` is the imaginary unit, ``C`` is the array `coeffs`,
``\\mathcal I`` is `CartesianIndices(C)`, ``\\vec{n}`` is a `CartesianIndex` and
``\\vec{k}_{\\vec{n}}`` is equal to ``n_j/p_j`` in the
``j``th position with ``p_j`` the ``j``th element of `period`.
Because of the choice to use Cartesian indices to set the phase factors,
typically the indices of `coeffs` should be specified by using an `OffsetArray`.
"""
struct FourierSeries{N,T,C,P} <: AbstractFourierSeries{N,T}
    coeffs::T
    period::P
    FourierSeries{N,T}(coeffs::C, period::P) where {N,T,C,P<:NTuple{N,<:Real}} =
        new{N,T,C,P}(coeffs, period)
end

FourierSeries(coeffs, period::Real) =
    FourierSeries(coeffs, ntuple(n -> period, Val(N)))
FourierSeries(coeffs::AbstractArray{T,N}, period::NTuple{N}) where {T,N} =
    FourierSeries{N}(coeffs, period)

value(f::FourierSeries{0}) = f.coeffs
period(f::FourierSeries) = f.period



"""
    FourierSeriesDerivative(f::FourierSeries{N}, a::NTuple{N}) where {N}

Represent the differential of Fourier series `f` by a multi-index `a` of
derivatives, e.g. `[1,2,...]`, whose `i`th entry represents the order of
differentiation on the `i`th input dimension of `f`. Mathematically, this means
```math
\\left( \\prod_{j=1}^N \\partial_{x_j}^{a_j} \\right) f(\\vec{x}) = \\sum_{\\vec{n} \\in \\mathcal I} \\left( \\prod_{j=1}^N (i 2\\pi k_j)^{a_j} \\right) C_{\\vec{n}} \\exp(i2\\pi\\vec{k}_{\\vec{n}}\\cdot\\overrightarrow{x})
```
where ``\\partial_{x_j}^{a_j}`` represents the ``a_j``th derivative of ``x_j``,
``i = \\sqrt{-1}`` is the imaginary unit, ``C`` is the array `coeffs`,
``\\mathcal I`` is `CartesianIndices(C)`, ``\\vec{n}`` is a `CartesianIndex` and
``\\vec{k}_{\\vec{n}}`` is equal to ``n_j/p_j`` in the ``j``th position with
``p_j`` the ``j``th element of `period`. Because of the choice to use Cartesian
indices to set the phase factors, typically the indices of `coeffs` should be
specified by using an `OffsetArray`. Also, note that termwise differentiation of
the Fourier series results in additional factors of ``i2\\pi`` which should be
anticipated for the use case. Also, note that this type can be used to represent
fractional differentiation or integration by suitably choosing the ``a_j``s.

This is a 'lazy' representation of the derivative because instead of
differentiating by computing all of the Fourier coefficients of the derivative
upon constructing the object, the evaluator waits until it contracts the
differentiated dimension to evaluate the new coefficients.
"""
struct FourierSeriesDerivative{N,T,F,A} <: AbstractFourierSeries{N,T}
    f::F
    a::A
    FourierSeriesDerivative{N,T}(f::F, a::A) where {N,T,F<:FourierSeries{N},A} =
        new{N,T,F,A}(f, a)
end

value(dv::FourierSeriesDerivative{0}) = value(dv.f)
period(dv::FourierSeriesDerivative) = period(dv.f)

"""
    OffsetFourierSeries(f::AbstractFourierSeries{N}, q::NTuple{N,Float64}) where {N}

Represent a Fourier series whose argument is offset by the vector ``\\vec{q}``
and evaluates it as ``f(\\vec{x}-\\vec{q})``.
"""
struct OffsetFourierSeries{N,T,F,Q} <: AbstractFourierSeries{N,T}
    f::T
    q::Q
    OffsetFourierSeries{N,T}(f::F, q::Q) where {N,T,F,Q} =
        new{N,T,F,Q}(f, q)
end
contract(f::OffsetFourierSeries, x::Number) = OffsetFourierSeries(contract(f.f, x-last(f.q)), pop(f.q))
period(f::OffsetFourierSeries) = period(f.f)
value(f::OffsetFourierSeries{0}) = value(f.f)

"""
    ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}

Represents a tuple of Fourier series of the same dimension and periodicity and
contracts them all simultaneously.
"""
struct ManyFourierSeries{N,T,F,P} <: AbstractFourierSeries{N,T}
    fs::T
    period::P
    ManyFourierSeries{N,T}(fs::F, period::P) where {N,T,F,P} =
        new{N,T,F,P}(fs, period)
end
function ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}
    @assert all(map(==(period(fs[1])), map(period, Base.tail(fs)))) "all periods should match"
    ManyFourierSeries(fs, period(fs[1]))
end
contract(fs::ManyFourierSeries, x::Number) = ManyFourierSeries(map(f -> contract(f, x), fs.fs), pop(fs.period))
period(fs::ManyFourierSeries) = fs.period
value(fs::ManyFourierSeries{0}) = map(value, fs.fs)

"""
    ManyOffsetsFourierSeries(f, qs..., [origin=true])

Represent a Fourier series evaluated at many different points, and contract them
all simultaneously, returning them in the order the `qs` were passed, i.e.
`(f(x-qs[1]), f(x-qs[2]), ...)`
The `origin` keyword decides whether or not to evaluate ``f`` without an offset,
and if `origin` is true, the value of ``f`` evaluated without an offset will be
returned in the first position of the output.
"""
struct ManyOffsetsFourierSeries{N,T,F,Q} <: AbstractFourierSeries{N,T}
    f::T
    qs::Q
    ManyOffsetsFourierSeries{N,T}(f::F, qs::Q) where {N,T,F,Q} =
        new{N,T,F,Q}(f, qs)
end

function ManyOffsetsFourierSeries(f::AbstractFourierSeries{N}, qs::NTuple{N,Float64}...; origin=true) where {N}
    qs_ = ifelse(origin, (fill(0.0, NTuple{N,Float64}),), ())
    ManyOffsetsFourierSeries(f, (qs_..., qs...))
end
function contract(f::ManyOffsetsFourierSeries, x::Number)
    fs = map(q -> OffsetFourierSeries(contract(f.f, x-last(q)), pop(q)), f.qs)
    ManyFourierSeries(fs, pop(period(f)))
end
period(f::ManyOffsetsFourierSeries) = period(f.f)
value(f::ManyOffsetsFourierSeries{0}) = value(f.f)
