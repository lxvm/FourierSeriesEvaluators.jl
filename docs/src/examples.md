# Examples

The following are several examples of how to use the Fourier series evaluators
exported by `FourierSeriesEvaluators` together with their documentation.

## [`FourierSeries`](@ref)

The `FourierSeries` constructor accepts an array of coefficients and a mandatory
keyword argument, `period`. To create a 3-dimensional series with random
coefficients we can use the following
```@repl
using FourierSeriesEvaluators
f = FourierSeries(rand(4,4,4), period=1.0)
```
We can then evaluate `f` at a random point using its function-like interface
`f(rand(3))`.

To control the indices of the coefficients, which determine the phase factors
according to a formula given later, we can either use arrays with offset
indices, or provide the offsets ourselves
```@repl
using FourierSeriesEvaluators, OffsetArrays
c = OffsetVector(rand(7), -3:3);
x = rand();
FourierSeries(c, period=1)(x) == FourierSeries(parent(c), period=1, offset=-4)(x)
```

If we want to take the derivative of a Fourier series such as the derivative of
sine, which is cosine, we can first construct and validate our version of `sin`
```@repl sincos
using FourierSeriesEvaluators
sine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2)
x = rand();
sine(x) ≈ sin(x)
```
and then reuse the arguments to the series and provide the order of derivative
with the `deriv` keyword, which is typically integer but can even be fractional
```@repl sincos
cosine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2, deriv=1)
cosine(x) ≈ cos(x)
```
For multidimensional series, a scalar value of `deriv` means it is applied to
derivatives of all variables, whereas a tuple or vector of orders will
select the order of derivative applied to each variable. For automatically
generating all orders of derivatives see the section on `DerivativeSeries`.


Lastly, if we wish to evaluate multiple Fourier series at the same time, we may
either use array-valued coefficients, such as with
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), or for very
large arrays we may pass a second positional argument to `FourierSeries` to
provide the number of variables in the `FourierSeries` corresponding to the
outermost axes of the coefficient array. Below we have an example constructing
equivalent evaluators using both styles.
```@repl
c = rand(5,7);
using StaticArrays, FourierSeriesEvaluators
sc = reinterpret(reshape, SVector{5,eltype(c)}, c);
x = rand();
FourierSeries(sc, period=1.0)(x) ≈ FourierSeries(c, 1, period=1.0)(x)
```
When using the form with the positional argument, note that inplace array
operations are used to avoid allocations, so be careful not to overwrite the
result if reusing it.

```@docs
FourierSeriesEvaluators.FourierSeries
```

## [`HermitianFourierSeries`](@ref)

In physics problems with periodicity, often is it possible to Fourier
interpolate Hermitian operators, such as in [`Wannier
interpolation`](https://en.wikipedia.org/wiki/Wannier_function). The
`HermitianFourierSeries` optimizes the interpolation by using the Hermitian
symmetry of the coefficients of such an interpolant, where ``C_{i} =
C_{-i}^{\dagger}``, resulting in a roughly 2x speedup and saving half the memory.

```@repl herm
using FourierSeriesEvaluators, OffsetArrays, StaticArrays
c = OffsetVector(rand(SMatrix{2,2,ComplexF64,4}, 7), -3:3);
c = c + reverse(adjoint.(c)); # symmetrize the coefficients
f = FourierSeries(c, period=1);
hf = HermitianFourierSeries(f)
```

As shown above, a `HermitianFourierSeries` can be constructed from an existing
`FourierSeries` whose coefficients satisfy the symmetry property.

```@repl herm
x = rand();
hf(x)
```

Observe that the output type is `SHermitianCompact`, which is a special type
that exploits the symmetry.

```@docs
FourierSeriesEvaluators.HermitianFourierSeries
```

## [`ManyFourierSeries`](@ref)

A third way to evaluate multiple series at the same time, possibly with
different types of coefficients, is with a `ManyFourierSeries`, which behaves
like a tuple of `FourierSeries`. We can construct one from multiple series and a
period (which the element series are rescaled by so that all periods match).
```@repl
using FourierSeriesEvaluators
f1 = FourierSeries(rand(3,3), period=3);
f2 = FourierSeries(rand(3,3), period=2);
x = rand(2);
ms = ManyFourierSeries(f1, f2, period=1)
ms(x)[1] == f1(3x)
ms(x)[2] == f2(2x)
```
There are situations in which inplace `FourierSeries` may be more efficient than
`ManyFourierSeries`, since the former calculates phase factors fewer times than
the later. 

```@docs
FourierSeriesEvaluators.ManyFourierSeries
```

## [`DerivativeSeries`](@ref)

To systematically compute all orders of derivatives of a Fourier series, we can
wrap a series with a `DerivativeSeries{O}`, where `O::Integer` specifies the
order of derivative to evaluate athe series and take all of its derivatives up
to and including degree `O`.
```@repl deriv
using FourierSeriesEvaluators
sine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2)
ds = DerivativeSeries{1}(sine)
ds(0)
```
We can use this to show that the fourth derivative of sine is itself
```@repl deriv
d4s = DerivativeSeries{4}(sine)
d4s(pi/3)
```
When applied to multidimensional series, all mixed partial derivatives are
computed exactly once and their layout in the tuple of results is explained
below.

```@docs
FourierSeriesEvaluators.DerivativeSeries
FourierSeriesEvaluators.JacobianSeries
FourierSeriesEvaluators.HessianSeries
```

## [`FourierWorkspace`](@ref)

By default, evaluating a Fourier series using the function-like interface
allocates several intermediate arrays in order to achieve the fastest-possible
evaluation with as few as possible calculations of phase factors. These arrays
can be preallocated in a `FourierWorkspace`
```@repl workspace
using FourierSeriesEvaluators
s = FourierSeries(rand(17,17,17), period=1);
x = ntuple(_->rand(), ndims(s));
ws = FourierSeriesEvaluators.workspace_allocate(s, Tuple(x));
@time s(x);
@time ws(x);
```
We can also allocate nested workspaces that can be used independently to
evaluate the same series at many points in a hierarchical or product grid in
parallel workloads.
```@repl workspace
ws3 = FourierSeriesEvaluators.workspace_allocate(s, Tuple(x), (2,2,2));
ws2 = FourierSeriesEvaluators.workspace_contract!(ws3, x[3], 1);
ws1 = FourierSeriesEvaluators.workspace_contract!(ws2, x[2], 2);
FourierSeriesEvaluators.workspace_evaluate!(ws1, x[1], 1) == s(x)
```
Note that the 3rd argument of `workspace_allocate`, `workspace_contract!`, and
`workspace_evaluate!` either specifies the number of nested workspaces to create
or selects the workspace to use.

```@autodocs
Modules = [FourierSeriesEvaluators]
Order   = [:type, :function]
Pages   = ["workspace.jl"]
```