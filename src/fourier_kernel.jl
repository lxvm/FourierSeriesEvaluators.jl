"""
    fourier_multiplier(a::Type, indices::Expr...)

Transform the index expressions `indices` to be exponentiated to obtain an
efficient Fourier multiplier that can be interpolated into the source code.

Optimizations of the Fourier multiplier(in the following let `i` be an index):
- `a === Val{0}`: returns `:nothing` expressions since there is no derivative
- `a === Val{1}`: returns `i` to avoid exponentiation
- `a <: Integer`: returns `i^a` to use integer exponent
Otherwise the fallback Fourier multiplier is `Complex(i)^a`
"""
function fourier_multiplier(a::Type, indices::Expr...)
    # codelets for derivatives as Fourier multipliers
    if a === Val{0} # elide the derivative altogether
        map(_ -> :true, indices)
    elseif a === Val{1} # elide the complex exponentiation in the derivative
        indices
    elseif a <: Integer # use integer exponentiation
        map(x -> :(($x)^a), indices)
    else # fallback Fourier multiplier formulas
        map(x -> :(Complex($x)^a), indices)
    end
end

"""
    nref_dim(N::Int, C::Symbol, i::Symbol, dim::Int, j::Union{Symbol,Expr})

Return an `N` dimensional indexing expression like `C[i_1,i_2,j]` or
`C[j,i_1,i_2]` where `N` controls the total number of dimensions, `dim` controls
the position of the inserted index `j`.
"""
function nref_dim(N::Int, C::Symbol, i::Symbol, dim::Int, j)
    :($C[$(ntuple(d -> d == dim ? j : Symbol(i, '_', d - (d >= dim)), N)...)])
end

"""
    cis_inv_op(x::Type, k::Type)

Return an expression with a function to invert `cis(x*k)` based on the type of
`x*k`. When `k*x isa Real` then `cis(x*k)` is a root of unity whose inverse is
its complex conjugate.
"""
cis_inv_op(x, k) = Base.promote_op(*, x, k) <: Real ? :(conj) : :(inv)

"""
    fourier_contract!(r::AbstractArray{T,N-1}, C::AbstractArray{T,N}, x, [k=1, a=0, shift=0, dim=Val(N)]) where {T,N}

Contract dimension `dim` of array `C` and write it to the array `r`, whose
axes must match `C`'s (excluding dimension `dim`). This function uses the indices in `axes(C,N)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series (see
[`fourier_multiplier`](@ref) for optimized values). The formula for what this
routine calculates is:
```math
r_{i_{1},\\dots,i_{N-1}} = \\sum_{i_{N}\\in\\text{axes}(C,N)} C_{i_{1},\\dots,i_{N-1},i_{N}+m+1} (ik (i_{N} + \\text{shift}))^{a} \\exp(ik x (i_{N} + \\text{shift}))
```
"""
@fastmath @generated function fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=oftype(x,1), a=Val(0), shift::Int=0, ::Val{dim}=Val(N)) where {R,T,N,N_,dim}
    N != N_+1 && return :(throw(ArgumentError("array dimensions incompatible")))
    R == fourier_type(T, x) || return :(throw(ArgumentError("result array of element type $R needs to store values of type $(fourier_type(T, x))")))
    # compute codelets based on input types
    ## indexing expressions with offset by dim
    r_i  = :(Base.Cartesian.@nref $N_ r i)
    C_i  = nref_dim(N, :C, :i, dim, :c)
    _C_i = nref_dim(N, :C, :i, dim, :(_c+n))
    C_i_ = nref_dim(N, :C, :i, dim, :(c_-n))
    ## optimized inverse of roots of unity
    zinv = cis_inv_op(x, k)
    ## 
    scalar_multiplier, m, _m, m_ =
        fourier_multiplier(a, :(im*k), :(c+shift), :(_c+n+shift), :(c_-n+shift))
    quote
        1 <= dim <= $N || throw(ArgumentError("selected dimension to contract is out of bounds"))
        Base.Cartesian.@nall $N_ d -> axes(r,d) == axes(C, d + (d >= dim)) ||
            throw(BoundsError("array axes incompatible (check size or indexing)"))

        s  = size(C, dim); isev = iseven(s)
        c  = firstindex(C, dim) # Find the index offset
        c += M = div(s, 2)      # translate to find axis center
        z  = cis(k*x*(c+shift)) # the initial offset phase
        z *= $scalar_multiplier # apply global Fourier multiplier (see expression defined above)

        # unroll first loop iteration
        b = !isev * z * $m      # obtain Fourier coefficient
        @inbounds Base.Cartesian.@nloops $N_ i r begin
            $r_i = b * $C_i
        end
        #= this loop could also be written as
        @inbounds r .= b .* selectdim(C, dim, c)
        =#

        s == 1 && return r      # fast return for singleton dimensions

        _dz = cis(k*x)          # forward--winding phase step
        dz_ = $(zinv)(_dz)       # backward-winding phase step

        z_ = z                  # initial winding phase
        _z = z * dz_^isev
        c_ = c                  # initial winding index
        _c = c - isev

        for n in Base.OneTo(M)  # symmetric winding loop
            _z *= _dz           # forward--winding phase
            z_ *= dz_           # backward-winding phase
            
            _b = _z * $_m       # forward--winding coefficient
            b_ = z_ * $m_       # backward-winding coefficient
            
            @inbounds Base.Cartesian.@nloops $N_ i r begin
                $r_i += _b * $_C_i + b_ * $C_i_
            end
            #= this loop could also be written as
            @inbounds r .+= _b .* selectdim(C, dim, _c+n) .+ b_ .* selectdim(C, dim, c_-n)
            =#
        end
        r
    end
end

"""
    fourier_contract(C::Vector, x, [k=1, a=0, shift=0, dim=Val(N)])

Identical to [`fourier_contract!`](@ref) except that it allocates its output.
"""
function fourier_contract(C::AbstractArray{T,N}, x, k=1, a=Val(0), shift=0, ::Val{dim}=Val(N)) where {T,N,dim}
    # make a copy of the uncontracted dimensions of C while preserving the axes
    # v = selectdim(C, dim, firstindex(C, dim)) # not type stable
    v = view(C, ntuple(n -> n==dim ? first(axes(C,n)) : axes(C,n), Val{N}())...)
    r = similar(v, fourier_type(T,x))
    fourier_contract!(r, C, x, k, a, shift, Val(dim))
end

"""
    fourier_evaluate(C::AbstractVector, x, [k=1, a=0, shift=0])

Evaluates a 1D Fourier series `C`. This function uses the indices in `axes(C,1)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series (see
[`fourier_multiplier`](@ref) for optimized values). The formula for what this
routine calculates is:
```math
r = \\sum_{i_{\\in\\text{axes}(C,1)} C_{i} (ik (i + \\text{shift}))^{a} \\exp(ik x (i + \\text{shift}))
```
"""
@fastmath @generated function fourier_evaluate(C::AbstractVector, x, k=oftype(x,1), a=Val(0), shift::Int=0)
    # compute codelets based on input types
    zinv = cis_inv_op(x, k)
    scalar_multiplier, m, _m, m_ =
        fourier_multiplier(a, :(im*k), :(c+shift), :(_c+n+shift), :(c_-n+shift))
    quote
        dim = 1
        s  = size(C, dim); isev = iseven(s)
        c  = firstindex(C, dim) # Find the index offset
        c += M = div(s, 2)      # translate to find axis center
        z  = cis(k*x*(c+shift)) # the initial offset phase
        z *= $scalar_multiplier # apply global Fourier multiplier (see expression defined above)

        b = !isev * z * $m      # obtain Fourier coefficient
        @inbounds r = b * C[c]  # unroll first loop iteration

        s == 1 && return r      # fast return for singleton dimensions
        
        _dz = cis(k*x)          # forward--winding phase step
        dz_ = $(zinv)(_dz)      # backward-winding phase step

        z_ = z                  # initial winding phase
        _z = z * dz_^isev
        c_ = c                  # initial winding index
        _c = c - isev

        for n in Base.OneTo(M)  # symmetric winding loop
            _z *= _dz           # forward--winding phase
            z_ *= dz_           # backward-winding phase
            
            _b = _z * $_m       # forward--winding coefficient
            b_ = z_ * $m_       # backward-winding coefficient
            
            @inbounds r += _b * C[_c+n] + b_ * C[c_-n]
        end
        r
    end
end

#= TODO: write a multidimensional evaluator with @nloops
@fastmath @generated function fourier_evaluate(C::AbstractArray{T,N}, x::NTuple{N}, k::NTuple{N}, a::NTuple{N}, shift=::NTuple{N}) where {T,N} end
=#