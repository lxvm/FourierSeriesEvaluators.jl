"""
    fourier_multiplier(a::Type, imk, init_index, frwd_index, bkwd_index)

Generate expressions `(def_outr_multiplier, def_init_multiplier,
def_wind_multiplier, outr_multiplier, init_multiplier, wind_multiplier)` that
are inserted into the source code to implement differentiation of Fourier series
computed by winding. The expressions `imk, init_index, frwd_index, bkwd_index,
z, init_term, frwd_term, bkwd_term` are substituted into the source code.

Optimizations of the Fourier multiplier(in the following let `i` be an index):
- `a === Val{0}`: returns `:nothing` expressions since there is no derivative
- `a === Val{1}`: returns `i` to avoid exponentiation
- `a <: Integer`: returns `i^a` to use integer exponent
Otherwise the fallback Fourier multiplier is `Complex(i)^a`
"""
function fourier_multiplier(a::Type, imk, init_index, frwd_index, bkwd_index, z, init_term, frwd_term, bkwd_term)
    # codelets for derivatives as Fourier multipliers
    if a === Val{0} # elide the derivative altogether
        def_outr_multiplier = def_init_multiplier = def_wind_multiplier =
            outr_multiplier = init_multiplier = wind_multiplier = :nothing
    else
        if a === Val{1} # elide the complex exponentiation in the derivative
            outr_multiplier = imk
            init_multiplier = init_index
            frwd_multiplier = frwd_index
            bkwd_multiplier = bkwd_index
        elseif a <: Integer # use integer exponentiation
            outr_multiplier = :(($imk)^a)
            init_multiplier = :(($init_index)^a)
            frwd_multiplier = :(($frwd_index)^a)
            bkwd_multiplier = :(($bkwd_index)^a)
        else # fallback Fourier multiplier formulas
            outr_multiplier = :(($imk)^a)
            init_multiplier = :(Complex($init_index)^a)
            frwd_multiplier = :(Complex($frwd_index)^a)
            bkwd_multiplier = :(Complex($bkwd_index)^a)
        end
        # define expressions to insert variables and multiplication into source code
        def_outr_multiplier = :(outr_multiplier = $outr_multiplier)
        def_init_multiplier = :(init_multiplier = $init_multiplier)
        def_wind_multiplier = quote
            frwd_multiplier = $frwd_multiplier
            bkwd_multiplier = $bkwd_multiplier
        end
        outr_multiplier = :($z *= outr_multiplier)
        init_multiplier = :($init_term *= init_multiplier)
        wind_multiplier = quote
            $frwd_term *= frwd_multiplier
            $bkwd_term *= bkwd_multiplier
        end
    end
    return (def_outr_multiplier, def_init_multiplier, def_wind_multiplier,
            outr_multiplier, init_multiplier, wind_multiplier)
end

"""
    cis_inv_op(x::Type, k::Type)

Return an expression with a function to invert `cis(x*k)` based on the type of
`x*k`. When `k*x isa Real` then `cis(x*k)` is a root of unity whose inverse is
its complex conjugate.
"""
cis_inv_op(x, k) = Base.promote_op(*, x, k) <: Real ? :(conj) : :(inv)

"""
    fourier_contract!(r::AbstractArray{T,N-1}, C::AbstractArray{T,N}, x, [k=1, a=0, shift=0]) where {T,N}

Contract the outermost index of array `C` and write it to the array `r`, whose
first `N-1` axes must match `C`'s. This function uses the indices in `axes(C,N)`
to evaluate the phase factors, which makes it compatible with `OffsetArray`s as
inputs. Optionally, a `shift` can be provided to manually offset the indices.
Also, `a` represents the order of derivative of the series (see
[`fourier_multiplier`](@ref) for optimized values). The formula for what this
routine calculates is:
```math
r_{i_{1},\\dots,i_{N-1}} = \\sum_{i_{N}\\in\\text{axes}(C,N)} C_{i_{1},\\dots,i_{N-1},i_{N}+m+1} (ik (i_{N} + \\text{shift}))^{a} \\exp(ik x (i_{N} + \\text{shift}))
```
"""
@fastmath @generated function fourier_contract!(r::AbstractArray{R,N_}, C::AbstractArray{T,N}, x, k=oftype(x,1), a=Val(0), shift::Int=0) where {R,T,N,N_}
    N != N_+1 && return :(throw(ArgumentError("array dimensions incompatible")))
    R == fourier_type(T, x) || return :(throw(ArgumentError("result array of element type $R needs to store values of type $(fourier_type(T, x))")))
    # define symbols that are substituted into codelets
    z, init_term, frwd_term, bkwd_term = :z, :init_term, :frwd_term, :bkwd_term
    # compute codelets based on input types
    zinv = cis_inv_op(x, k)
    def_outr_multiplier, def_init_multiplier, def_wind_multiplier, outr_multiplier, init_multiplier, wind_multiplier =
        fourier_multiplier(a, :(im*k), :(c+shift), :(_c+n+shift), :(c_-n+shift), z, init_term, frwd_term, bkwd_term)
    quote
        axes(r) == axes(C)[1:$N_] || throw(ArgumentError("array axes incompatible (check size or indexing)"))
        
        s  = size(C, $N)
        c  = first(axes(C, $N)) # Find the index offset
        c += m = div(s, 2)      # translate to find axis center
        $z = cis(k*x*(c+shift)) # the initial offset phase
        $def_outr_multiplier    # define global Fourier multiplier (see expression defined above)
        $outr_multiplier        # apply global Fourier multiplier (see expression defined above)

        dz   = cis(k*x)         # forward--winding phase step
        dz⁻¹ = $(zinv)(dz)      # backward-winding phase step

        _z = z_ = $z            # initial winding phase
        _c = c_ = c             # initial winding index
        
        # adjust initial values for winding loop
        isev = iseven(s)
        isod = !isev
        _z *= dz⁻¹^isev
        _c -= isev

        # unroll first loop iteration
        $def_init_multiplier
        @inbounds Base.Cartesian.@nloops $N_ i r begin
            $init_term = isod * $z * (Base.Cartesian.@nref $N C d -> d == $N ? c : i_d)
            $init_multiplier
            (Base.Cartesian.@nref $N_ r i) = $init_term
        end

        # symmetric winding loop
        for n in Base.OneTo(m)
            _z *= dz   # forward--winding coefficient
            z_ *= dz⁻¹ # backward-winding coefficient
            
            $def_wind_multiplier # define Fourier multipliers (see expression above)

            @inbounds Base.Cartesian.@nloops $N_ i r begin
                $frwd_term = _z * (Base.Cartesian.@nref $N C d -> d == $N ? _c+n : i_d)
                $bkwd_term = z_ * (Base.Cartesian.@nref $N C d -> d == $N ? c_-n : i_d)
                $wind_multiplier
                (Base.Cartesian.@nref $N_ r i) += $frwd_term + $bkwd_term
            end
        end
        r
    end
end

"""
    fourier_contract(C::Vector, x, [k=1, a=0, shift=0])

Identical to [`fourier_contract!`](@ref) except that it allocates its output.
"""
function fourier_contract(C::AbstractArray{T,N}, x, k=1, a=Val(0), shift=0) where {T,N}
    # make a copy of the first N-1 dimensions of C while preserving the axes
    ax = ntuple(n -> n==N ? first(axes(C,N)) : axes(C,n), Val{N}())
    r = similar(view(C, ax...), fourier_type(T,x))
    fourier_contract!(r, C, x, k, a, shift)
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
    # define symbols that are substituted into codelets
    z, init_term, frwd_term, bkwd_term = :z, :init_term, :frwd_term, :bkwd_term
    # compute codelets based on input types
    zinv = cis_inv_op(x, k)
    def_outr_multiplier, def_init_multiplier, def_wind_multiplier, outr_multiplier, init_multiplier, wind_multiplier =
        fourier_multiplier(a, :(im*k), :(c+shift), :(_c+n+shift), :(c_-n+shift), z, init_term, frwd_term, bkwd_term)
    quote
        s  = size(C, 1)
        c  = first(axes(C, 1)) # Find the index offset
        c += m = div(s, 2)      # translate to find axis center
        $z = cis(k*x*(c+shift)) # the initial offset phase
        $def_outr_multiplier    # define global Fourier multiplier (see expression defined above)
        $outr_multiplier        # apply global Fourier multiplier (see expression defined above)

        dz   = cis(k*x)         # forward--winding phase step
        dz⁻¹ = $(zinv)(dz)      # backward-winding phase step

        _z = z_ = $z            # initial winding phase
        _c = c_ = c             # initial winding index
        
        # adjust initial values for winding loop
        isev = iseven(s)
        isod = !isev
        _z *= dz⁻¹^isev
        _c -= isev

        # unroll first loop iteration
        $def_init_multiplier
        $init_term = isod * $z * C[c]
        $init_multiplier
        r = $init_term

        # symmetric winding loop
        for n in Base.OneTo(m)
            _z *= dz   # forward--winding coefficient
            z_ *= dz⁻¹ # backward-winding coefficient
            
            $def_wind_multiplier # define Fourier multipliers (see expression above)

            $frwd_term = _z * C[_c+n]
            $bkwd_term = z_ * C[c_-n]
            $wind_multiplier
            r += $frwd_term + $bkwd_term
        end
        r
    end
end

#= TODO: write a multidimensional evaluator with @nloops
@fastmath @generated function fourier_evaluate(C::AbstractArray{T,N}, x::NTuple{N}, k::NTuple{N}, a::NTuple{N}, shift=::NTuple{N}) where {T,N} end
=#