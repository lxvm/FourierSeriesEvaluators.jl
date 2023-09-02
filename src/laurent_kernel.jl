istwo(x) = x == (one(x) + one(x))

# goal: evaluate the Laurent series, e.g. 1d: sum(C[n] * (n+o)^a * z^(n-a) for n in eachindex(C))
@fastmath function laurent_evaluate_(C::AbstractArray{T,N}, zs::NTuple{N,F}, us::NTuple{N,F}, as::NTuple{N,Number}, os::NTuple{N,Integer}) where {T,N,F<:Complex{<:AbstractFloat}}
    N == 0 && return C[]

    zz, z = zs[1:N-1], zs[N]
    uu, u = us[1:N-1], us[N]
    aa, a = as[1:N-1], as[N]
    oo, o = os[1:N-1], os[N]

    s = size(C, N); isev = iseven(s)
    c = firstindex(C, N)    # Find the index offset
    c += M = div(s, 2)      # translate to find axis center

    # inital Laurent coefficient
    z0 = one(u) # u^a is correct, but expensive to evaluate so we factor it out of the sum

    # unroll first loop iteration
    b = isev ? zero(z0) : (iszero(a) ? z0 : z0*(c+o)^a)
    if N == 1
        @inbounds r = b * C[c]
    else
        # r = b * laurent_evaluate_(selectdim(C, N, c), zz, uu, aa, oo)
        # we can save 10 ns by constructing the view manually instead of with selectdim
        r = b * laurent_evaluate_(view(C, ntuple(n -> (:), Val(N-1))..., c), zz, uu, aa, oo)
    end
    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = z, u         # forward-backward-winding polynomial steps

    z_, c_ = z0, c          # initial winding polynomial, index
    _z, _c = isev ? (z0*dz_, c-1) : (z0, c)

    for n in 1:M            # symmetric winding loop
        _z *= _dz           # forward--winding polynomial
        z_ *= dz_           # backward-winding polynomial

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        # winding coefficients
        _b, b_ = _iszero(a) ? (_z, z_) : (_z*(_i+o)^a, z_*(i_+o)^a)

        if N == 1
            @inbounds r += _b * C[_i] + b_ * C[i_]
        else
            # r += _b * laurent_evaluate_(selectdim(C, N, _i), zz, uu, aa, oo) +
            #      b_ * laurent_evaluate_(selectdim(C, N, i_), zz, uu, aa, oo)
            r += _b * laurent_evaluate_(view(C, ntuple(n -> (:), Val(N-1))..., _i), zz, uu, aa, oo) +
                 b_ * laurent_evaluate_(view(C, ntuple(n -> (:), Val(N-1))..., i_), zz, uu, aa, oo)

        end
    end
    return r
end

function laurent_evaluate(
    C::AbstractArray{T,N},
    zs::NTuple{N,F},
    us::NTuple{N,F}=map(inv, zs),
    as::NTuple{N,Number}=ntuple(zero, Val(N)),
    os::NTuple{N,Integer}=ntuple(zero, Val(N)),
) where {T,N,F<:Complex{<:AbstractFloat}}
    return mapreduce(^, *, us, as) * laurent_evaluate_(C, zs, us, as, os)
end
