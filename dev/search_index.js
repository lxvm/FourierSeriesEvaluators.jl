var documenterSearchIndex = {"docs":
[{"location":"methods/#Manual","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"methods/","page":"Manual","title":"Manual","text":"Modules = [FourierSeriesEvaluators]\nOrder   = [:type, :function]","category":"page"},{"location":"methods/#FourierSeriesEvaluators.AbstractFourierSeries","page":"Manual","title":"FourierSeriesEvaluators.AbstractFourierSeries","text":"AbstractFourierSeries{N,T}\n\nA supertype for Fourier series that are periodic maps R^N to V where V is any vector space with elements of type T. Typically these can be represented by N-dimensional arrays whose elements belong to the vector space.\n\n(f::AbstractFourierSeries)(x) = evaluate(f, promote(x...))\n\nEvaluate the Fourier series at the given point (see evaluate).\n\n\n\n\n\n","category":"type"},{"location":"methods/#FourierSeriesEvaluators.AbstractInplaceFourierSeries","page":"Manual","title":"FourierSeriesEvaluators.AbstractInplaceFourierSeries","text":"AbstractInplaceFourierSeries{N,T} <: AbstractFourierSeries{N,T}\n\nA supertype for Fourier series evaluated in place. These define the contract! method instead of contract.\n\n\n\n\n\n","category":"type"},{"location":"methods/#FourierSeriesEvaluators.FourierSeries","page":"Manual","title":"FourierSeriesEvaluators.FourierSeries","text":"FourierSeries(coeffs::AbstractArray; period=2pi, offset=0, deriv=0, shift=0)\n\nConstruct a Fourier series whose coefficients are given by the coefficient array array coeffs whose elements should support addition and scalar multiplication, This type represents the Fourier series\n\nf(vecx) = sum_vecn in mathcal I C_vecn exp(i2piveck_vecncdotoverrightarrowx)\n\nThe indices vecn are the CartesianIndices of coeffs. Also, the keywords, which can either be a single value applied to all dimensions or a collection describing each dimension mean\n\nperiod: The periodicity of the Fourier series. Equivalent to 2pik\noffset: An offset in the phase index, which must be integer\nderiv: The degree of differentiation, implemented as a Fourier multiplier\nshift: A translation q such that the evaluation point x is shifted to x-q\n\n\n\n\n\n","category":"type"},{"location":"methods/#FourierSeriesEvaluators.InplaceFourierSeries","page":"Manual","title":"FourierSeriesEvaluators.InplaceFourierSeries","text":"InplaceFourierSeries(coeffs::AbstractArray; period=2pi, offset=0, deriv=0, shift=0)\n\nSimilar to FourierSeries except that it doesn't allocate new arrays for every call to contract and contract is limited to the outermost dimension/variable of the series.\n\n\n\n\n\n","category":"type"},{"location":"methods/#FourierSeriesEvaluators.ManyFourierSeries","page":"Manual","title":"FourierSeriesEvaluators.ManyFourierSeries","text":"ManyFourierSeries(fs::AbstractFourierSeries{N}...) where {N}\n\nRepresents a tuple of Fourier series of the same dimension and periodicity and contracts them all simultaneously.\n\n\n\n\n\n","category":"type"},{"location":"methods/#Base.eltype-Union{Tuple{Type{var\"#s4\"} where var\"#s4\"<:AbstractFourierSeries{N, T}}, Tuple{T}, Tuple{N}} where {N, T}","page":"Manual","title":"Base.eltype","text":"eltype(::AbstractFourierSeries{N,T}) where {N,T}\n\nReturns T, the type of the input data to the Fourier series. For the output type, see fourier_type\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.cis_inv_op-Tuple{Any, Any}","page":"Manual","title":"FourierSeriesEvaluators.cis_inv_op","text":"cis_inv_op(x::Type, k::Type)\n\nReturn an expression with a function to invert cis(x*k) based on the type of x*k. When k*x isa Real then cis(x*k) is a root of unity whose inverse is its complex conjugate.\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.coefficients-Tuple{AbstractFourierSeries}","page":"Manual","title":"FourierSeriesEvaluators.coefficients","text":"coefficients(f::AbstractFourierSeries) = nothing\n\nReturn the underlying array representing the Fourier series, if applicable. There are cases with multiple Fourier series where this many not make sense.\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.contract","page":"Manual","title":"FourierSeriesEvaluators.contract","text":"contract(f::AbstractFourierSeries{N}, x::Number, ::Val{dim}) where {N,dim}\n\nReturn another Fourier series of dimension N-1 by summing over dimension dim of f with the phase factors evaluated at x.\n\nnote: For developers\nImplementations of the interface need to provide an implementation of contract with the same signature as the above that specialize on the concrete type of f. While any value of dim from 1 to N is can be implemented, it is most important to implement dim==N.\n\n\n\n\n\n","category":"function"},{"location":"methods/#FourierSeriesEvaluators.contract!","page":"Manual","title":"FourierSeriesEvaluators.contract!","text":"contract!(f::AbstractInplaceFourierSeries, x::Number, dim::Type{Val{d}})\n\nAn in-place version of contract, however the argument dim must be a Val{d} in order to dispatch to the specific contract method. This should return f.\n\nnote: For developers\nAn AbstractInplaceFourierSeries only needs to implement contract!, which is set up to be called by contract\n\n\n\n\n\n","category":"function"},{"location":"methods/#FourierSeriesEvaluators.evaluate","page":"Manual","title":"FourierSeriesEvaluators.evaluate","text":"evaluate(f::AbstractFourierSeries, x)\n\nEvaluate the Fourier series at the point x. By default x is wrapped into a tuple and the Fourier series is contracted along the outer dimension.\n\nnote: For developers\nImplementations of the interface only need to define a method specializing on the concrete type T of f::T with signature evaluate(T, ::NTuple{1}) while evaluation of the other dimensions can be delegated to contract.\n\n\n\n\n\n","category":"function"},{"location":"methods/#FourierSeriesEvaluators.fourier_contract!-Union{Tuple{dim}, Tuple{N_}, Tuple{N}, Tuple{T}, Tuple{R}, Tuple{AbstractArray{R, N_}, AbstractArray{T, N}, Any}, Tuple{AbstractArray{R, N_}, AbstractArray{T, N}, Any, Any}, Tuple{AbstractArray{R, N_}, AbstractArray{T, N}, Any, Any, Any}, Tuple{AbstractArray{R, N_}, AbstractArray{T, N}, Any, Any, Any, Int64}, Tuple{AbstractArray{R, N_}, AbstractArray{T, N}, Any, Any, Any, Int64, Val{dim}}} where {R, T, N, N_, dim}","page":"Manual","title":"FourierSeriesEvaluators.fourier_contract!","text":"fourier_contract!(r::AbstractArray{T,N-1}, C::AbstractArray{T,N}, x, [k=1, a=0, shift=0, dim=Val(N)]) where {T,N}\n\nContract dimension dim of array C and write it to the array r, whose axes must match C's (excluding dimension dim). This function uses the indices in axes(C,N) to evaluate the phase factors, which makes it compatible with OffsetArrays as inputs. Optionally, a shift can be provided to manually offset the indices. Also, a represents the order of derivative of the series (see fourier_multiplier for optimized values). The formula for what this routine calculates is:\n\nr_i_1dotsi_N-1 = sum_i_Nintextaxes(CN) C_i_1dotsi_N-1i_N+m+1 (ik (i_N + textshift))^a exp(ik x (i_N + textshift))\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.fourier_contract-Union{Tuple{dim}, Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, N}, Any}, Tuple{AbstractArray{T, N}, Any, Any}, Tuple{AbstractArray{T, N}, Any, Any, Any}, Tuple{AbstractArray{T, N}, Any, Any, Any, Any}, Tuple{AbstractArray{T, N}, Any, Any, Any, Any, Val{dim}}} where {T, N, dim}","page":"Manual","title":"FourierSeriesEvaluators.fourier_contract","text":"fourier_contract(C::Vector, x, [k=1, a=0, shift=0, dim=Val(N)])\n\nIdentical to fourier_contract! except that it allocates its output.\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.fourier_evaluate","page":"Manual","title":"FourierSeriesEvaluators.fourier_evaluate","text":"fourier_evaluate(C::AbstractVector, x, [k=1, a=0, shift=0])\n\nEvaluates a 1D Fourier series C. This function uses the indices in axes(C,1) to evaluate the phase factors, which makes it compatible with OffsetArrays as inputs. Optionally, a shift can be provided to manually offset the indices. Also, a represents the order of derivative of the series (see fourier_multiplier for optimized values). The formula for what this routine calculates is:\n\nr = sum_i_intextaxes(C1) C_i (ik (i + textshift))^a exp(ik x (i + textshift))\n\n\n\n\n\n","category":"function"},{"location":"methods/#FourierSeriesEvaluators.fourier_multiplier-Tuple{Type, Vararg{Expr, N} where N}","page":"Manual","title":"FourierSeriesEvaluators.fourier_multiplier","text":"fourier_multiplier(a::Type, indices::Expr...)\n\nTransform the index expressions indices to be exponentiated to obtain an efficient Fourier multiplier that can be interpolated into the source code.\n\nOptimizations of the Fourier multiplier(in the following let i be an index):\n\na === Val{0}: returns :nothing expressions since there is no derivative\na === Val{1}: returns i to avoid exponentiation\na <: Integer: returns i^a to use integer exponent\n\nOtherwise the fallback Fourier multiplier is Complex(i)^a\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.fourier_type-Union{Tuple{T}, Tuple{Type{T}, Any}} where T","page":"Manual","title":"FourierSeriesEvaluators.fourier_type","text":"fourier_type(::Type{T}, x) where T = Base.promote_op(*, T, phase_type(x))\nfourier_type(C::AbstractFourierSeries, x) = fourier_type(eltype(f), x)\n\nReturns the output type of the Fourier series.\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.nref_dim-Tuple{Int64, Symbol, Symbol, Int64, Any}","page":"Manual","title":"FourierSeriesEvaluators.nref_dim","text":"nref_dim(N::Int, C::Symbol, i::Symbol, dim::Int, j::Union{Symbol,Expr})\n\nReturn an N dimensional indexing expression like C[i_1,i_2,j] or C[j,i_1,i_2] where N controls the total number of dimensions, dim controls the position of the inserted index j.\n\n\n\n\n\n","category":"method"},{"location":"methods/#FourierSeriesEvaluators.period","page":"Manual","title":"FourierSeriesEvaluators.period","text":"period(f::AbstractFourierSeries{N}) where {N}\n\nReturn a NTuple{N} whose m-th element corresponds to the period of f along its m-th input dimension. Typically, these values set the units of length for the problem.\n\n\n\n\n\n","category":"function"},{"location":"methods/#FourierSeriesEvaluators.phase_type-Tuple{Any}","page":"Manual","title":"FourierSeriesEvaluators.phase_type","text":"phase_type(x) = Base.promote_op(cis, eltype(x))\n\nReturns the type of exp(im*x).\n\n\n\n\n\n","category":"method"},{"location":"#FourierSeriesEvaluators.jl","page":"Home","title":"FourierSeriesEvaluators.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"FourierSeriesEvaluators","category":"page"},{"location":"#FourierSeriesEvaluators","page":"Home","title":"FourierSeriesEvaluators","text":"A package implementing fast, multi-dimensional Fourier series evaluators that are more convenient than FFTs when using hierarchical grids and the number of coefficients is small compared to the number of evaluation points.\n\nFor example, to evaluate cosine\n\nFourierSeries([0.5, 0.0, 0.5], period=2pi, offset=-2)(pi) ≈ cos(pi)\n\nor to evaluate sine from the derivative of cosine\n\nFourierSeries([0.5, 0.0, 0.5], period=2pi, offset=-2, deriv=1)(pi/2) ≈ -sin(pi/2)\n\nThe package also provides the following low-level routines that are also useful\n\nfourier_contract: contracts 1 index of a multidimensional Fourier series\nfourier_evaluate: evaluates 1d Fourier series\n\nThese routines have the following features\n\nSupport for abstract (esp. offset) arrays\nSupport for evaluation in the complex plane\nEvaluation of derivatives of the Fourier series with Fourier multipliers\n\n\n\n\n\n","category":"module"}]
}
