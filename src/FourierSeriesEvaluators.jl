"""
A package implementing fast, multi-dimensional Fourier series evaluators that
are more convenient than FFTs when using hierarchical grids and the number of
coefficients is small compared to the number of evaluation points.

For example, to evaluate cosine

    FourierSeries([0.0, 0.0, 0.5], period=2pi, offset=-2)(pi) ≈ cos(pi)

The package also provides the following low-level routines that are also useful
- `fourier_contract`: contracts 1 index of a multidimensional Fourier series
- `fourier_evaluate`: evaluates 1d Fourier series
These routines have the following features
- Support for abstract (esp. offset) arrays
- Support for evaluation in the complex plane
- Evaluation of derivatives of the Fourier series with Fourier multipliers
"""
module FourierSeriesEvaluators

using AbstractFourierSeriesEvaluators: AbstractFourierSeries, AbstractInplaceFourierSeries, phase_type, fourier_type
import AbstractFourierSeriesEvaluators: period, contract, contract!, evaluate

export AbstractFourierSeries, AbstractInplaceFourierSeries, phase_type, fourier_type,
    period, contract, contract!, evaluate

export fourier_contract, fourier_contract!, fourier_evaluate
include("fourier_kernel.jl")

export FourierSeries, ManyFourierSeries
include("FourierSeries.jl")

export FourierSeries3D
include("FourierSeries3D.jl")

end