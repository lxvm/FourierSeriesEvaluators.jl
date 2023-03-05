# FourierSeriesEvaluators.jl

[Documentation](https://lxvm.github.io/FourierSeriesEvaluators.jl/dev/)

This package provides multi-dimensional Fourier series evaluators for adaptive
grid evaluation. They accept arbitrary array types as coefficients and support
the `OffsetArray` type.

These Fourier series are fast to evaluate in the sense that they have been
carefully optimized. The algorithm they implement is equivalent to multiplying a
row of a DFT matrix against the column vector of coefficients. Thus this code is
preferrable to FFTs only when the number of evaluation points is somewhat larger
than the number of Fourier coefficients, or when evaluation on non-equispace
grids is needed.

## Author and Copyright

FourierSeriesEvaluators.jl was written by [Lorenzo Van
Muñoz](https://web.mit.edu/lxvm/www/), and is free/open-source software under
the MIT license.

## Related packages
- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl)
- [OffsetArrays.jl](https://github.com/JuliaArrays/OffsetArrays.jl)
