# FourierSeriesEvaluators.jl

| Documentation | Build Status | Coverage | Version |
| :-: | :-: | :-: | :-: |
| [![][docs-stable-img]][docs-stable-url] | [![][action-img]][action-url] | [![][codecov-img]][codecov-url] | [![ver-img]][ver-url] |
| [![][docs-dev-img]][docs-dev-url] | [![][pkgeval-img]][pkgeval-url] | [![][aqua-img]][aqua-url] | [![deps-img]][deps-url] |


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


<!-- badges -->

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lxvm.github.io/FourierSeriesEvaluators.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lxvm.github.io/FourierSeriesEvaluators.jl/dev/

[action-img]: https://github.com/lxvm/FourierSeriesEvaluators.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/lxvm/FourierSeriesEvaluators.jl/actions/?query=workflow:CI

[pkgeval-img]: https://juliahub.com/docs/General/FourierSeriesEvaluators/stable/pkgeval.svg
[pkgeval-url]: https://juliahub.com/ui/Packages/General/FourierSeriesEvaluators

[codecov-img]: https://codecov.io/github/lxvm/FourierSeriesEvaluators.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/github/lxvm/FourierSeriesEvaluators.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[ver-img]: https://juliahub.com/docs/FourierSeriesEvaluators/version.svg
[ver-url]: https://juliahub.com/ui/Packages/FourierSeriesEvaluators/UDEDl

[deps-img]: https://juliahub.com/docs/General/FourierSeriesEvaluators/stable/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/General/FourierSeriesEvaluators?t=2