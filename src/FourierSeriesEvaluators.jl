"""
A package implementing efficient, multi-dimensional Fourier interpolation that is more
convenient than FFTs when using hierarchical grids and when the number of coefficients is
small compared to the number of evaluation points.

### Quickstart

As a first example, we evaluate cosine from its Fourier series coefficients:

    julia> using FourierSeriesEvaluators

    julia> cosine = FourierSeries([0.5, 0.0, 0.5], period=2pi, offset=-2)
    3-element and (6.283185307179586,)-periodic FourierSeries with Float64 coefficients, (0,) derivative, (-2,) offset

    julia> cosine(pi)
    -1.0 + 0.0im

Notice that we can create a series object and interpolate with a function-like interface.
For more examples, see the documentation.

### Features

Generic Fourier series can be implemented with the [`AbstractFourierSeries`](@ref) interface
and the following implementations are provided as building blocks for others:
- [`FourierSeries`](@ref): (inplace) evaluation at real/complex arguments
- [`ManyFourierSeries`](@ref): evaluation of multiple series
- [`DerivativeSeries`](@ref): automatic evaluation of derivatives of series to any order

Additionally, memory management for parallel workloads is supported by
[`FourierWorkspace`](@ref).

# Extended help

The package also provides the following low-level routines that are also useful
- [`fourier_contract`](@ref): contracts 1 variable of a multidimensional Fourier series
- [`fourier_evaluate`](@ref): evaluates N-dimensional Fourier series with no allocations
These routines have the following features
- Support for abstract (esp. offset) coefficient arrays and array-valued coefficients
- Support for evaluation in the complex plane
- Evaluation of derivatives of the Fourier series with Fourier multipliers
"""
module FourierSeriesEvaluators

export AbstractFourierSeries
include("definitions.jl")

export fourier_contract, fourier_contract!, fourier_evaluate, fourier_allocate
include("fourier_kernel.jl")

export FourierSeries, ManyFourierSeries, DerivativeSeries, JacobianSeries, HessianSeries
include("fourier_series.jl")

export FourierWorkspace
include("workspace.jl")

end
