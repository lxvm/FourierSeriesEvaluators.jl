# FourierSeriesEvaluators.jl

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/lxvm/FourierSeriesEvaluators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lxvm/FourierSeriesEvaluators.jl/actions/?query=workflow:CI)
[![Documentation - in development](https://img.shields.io/badge/docs-main-blue.svg)](https://lxvm.github.io/FourierSeriesEvaluators.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


This package provides multi-dimensional Fourier interpolants for arrays of
Fourier coefficients. Its documented features include:
- Support for arbitrary coefficient types and coefficients with offset indices
- Inplace evaluation of (multiple) series at real and complex arguments
- Arbitrary orders of derivatives of series
- Memory management intended for parallel workloads

These Fourier series are efficient to evaluate in the sense that they have been
carefully optimized. The algorithm they implement is equivalent to multiplying a
row of a DFT matrix against the column vector of coefficients. Thus this code is
preferrable to FFTs only when the number of evaluation points is somewhat larger
than the number of Fourier coefficients, or when interpolation on non-equispace
grids is needed.

## Author and Copyright

FourierSeriesEvaluators.jl was written by [Lorenzo Van
Muñoz](https://web.mit.edu/lxvm/www/), and is free/open-source software under
the MIT license.

## Related packages
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl)
