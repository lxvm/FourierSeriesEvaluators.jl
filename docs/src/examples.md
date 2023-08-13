# Examples

The following are several examples of how to use the Fourier series evaluators
exported by `FourierSeriesEvaluators` together with their documentation.

## [`FourierSeries`](@ref)

The `FourierSeries` constructor accepts an array of coefficients and a mandatory
keyword argument, `period`. To create a 3-dimensional series with random
coefficients we can use the following
```
julia> f = FourierSeries(rand(4,4,4), period=1.0)
4×4×4 and (1.0, 1.0, 1.0)-periodic FourierSeries with Float64 coefficients, (0, 0, 0) derivative, (0, 0, 0) offset
```
We can then evaluate `f` at a random point using its function-like interface
`f(rand(3))`.

To control the indices of the coefficients, which determine the phase factors
according to a formula given later, we can either use arrays with offset
indices, or provide the offsets ourselves
```
julia> using OffsetArrays

julia> c = OffsetVector(rand(7), -3:3);

julia> x = rand();

julia> FourierSeries(c, period=1)(x) == FourierSeries(parent(c), period=1, offset=-4)(x)
true
```

If we want to take the derivative of a Fourier series such as the derivative of
sine, which is cosine, we can first construct and validate our version of `sin`
```
julia> sine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2)
3-element and (6.283185307179586,)-periodic FourierSeries with ComplexF64 coefficients, (0,) derivative, (-2,) offset

julia> x = rand();

julia> sine(x) ≈ sin(x)
true
```
and then reuse the arguments to the series and provide the order of derivative
with the `deriv` keyword, which is typically integer but can even be fractional
```
julia> cosine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2, deriv=1)
3-element and (6.283185307179586,)-periodic FourierSeries with ComplexF64 coefficients, (1,) derivative, (-2,) offset

julia> cosine(x) ≈ cos(x)
true
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
```
julia> c = rand(5,7);

julia> using StaticArrays

julia> sc = reinterpret(reshape, SVector{5,eltype(c)}, c);

julia> x = rand();

julia> FourierSeries(sc, period=1.0)(x) ≈ FourierSeries(c, 1, period=1.0)(x)
true
```
When using the form with the positional argument, note that inplace array
operations are used to avoid allocations, so be careful not to overwrite the
result if reusing it.

```@docs
FourierSeriesEvaluators.FourierSeries
```

## [`ManyFourierSeries`](@ref)

A third way to evaluate multiple series at the same time, possibly with
different types of coefficients, is with a `ManyFourierSeries`, which behaves
like a tuple of `FourierSeries`. We can construct one from multiple series and a
period (which the element series are rescaled by so that all periods match).
```
julia> f1 = FourierSeries(rand(3,3), period=3);

julia> f2 = FourierSeries(rand(3,3), period=2);

julia> x = rand(2);

julia> ms = ManyFourierSeries(f1, f2, period=1)
2-dimensional and (1, 1)-periodic ManyFourierSeries with 2 series

julia> ms(x)[1] == f1(3x)
true

julia> ms(x)[2] == f2(2x)
true
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
```
julia> sine = FourierSeries([-1, 0, 1]/2im, period=2pi, offset=-2)
3-element and (6.283185307179586,)-periodic FourierSeries with ComplexF64 coefficients, (0,) derivative, (-2,) offset

julia> ds = DerivativeSeries{1}(sine)
1-dimensional and (6.283185307179586,)-periodic DerivativeSeries of order 1

julia> ds(0)
(0.0 + 0.0im, (1.0 + 0.0im,))
```
We can use this to show that the fourth derivative of sine is itself
```
julia> d4s = DerivativeSeries{4}(sine)
1-dimensional and (6.283185307179586,)-periodic DerivativeSeries of order 4

julia> d4s(pi/3)
(0.8660254037844386 + 0.0im, (0.5000000000000001 + 0.0im,), ((-0.8660254037844386 + 0.0im,),), (((-0.5000000000000001 + 0.0im,),),), ((((0.8660254037844386 + 0.0im,),),),))
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
```
julia> s = FourierSeries(rand(17,17,17), period=1)
17×17×17 and (1, 1, 1)-periodic FourierSeries with Float64 coefficients, (0, 0, 0) derivative, (0, 0, 0) offset

julia> ws = FourierSeriesEvaluators.workspace_allocate(s, Tuple(x));

julia> @time s(x);
  0.000044 seconds (3 allocations: 5.047 KiB)

julia> @time ws(x);
  0.000063 seconds (1 allocation: 32 bytes)
```
We can also allocate nested workspaces that can be used independently to
evaluate the same series at many points in a hierarchical or product grid in
parallel workloads.
```
julia> ws3 = FourierSeriesEvaluators.workspace_allocate(s, Tuple(x), (2,2,2));

julia> ws2 = FourierSeriesEvaluators.workspace_contract!(ws3, x[3], 1);

julia> ws1 = FourierSeriesEvaluators.workspace_contract!(ws2, x[2], 2);

julia> FourierSeriesEvaluators.workspace_evaluate!(ws1, x[1], 1) == s(x)
true
```
Note that the 3rd argument of `workspace_allocate`, `workspace_contract!`, and
`workspace_evaluate!` either specifies the number of nested workspaces to create
or selects the workspace to use.

```@autodocs
Modules = [FourierSeriesEvaluators]
Order   = [:type, :function]
Pages   = ["workspace.jl"]
```