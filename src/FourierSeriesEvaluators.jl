"""
A package implementing fast, multi-dimensional Fourier series evaluators that
are more convenient than FFTs when using hierarchical grids and the number of
coefficients is small compared to the number of evaluation points. For examples
see the tests.
"""
module FourierSeriesEvaluators

using AbstractFourierSeriesEvaluators: AbstractFourierSeries, AbstractInplaceFourierSeries, phase_type, fourier_type
import AbstractFourierSeriesEvaluators: period, contract, contract!, value

export AbstractFourierSeries, AbstractInplaceFourierSeries, phase_type, fourier_type,
    period, contract, contract!, value

export fourier_kernel
include("fourier_kernel.jl")

export FourierSeries, FourierSeriesDerivative, OffsetFourierSeries, ManyFourierSeries, ManyOffsetsFourierSeries
include("FourierSeries.jl")

export FourierSeries3D
include("FourierSeries3D.jl")

end