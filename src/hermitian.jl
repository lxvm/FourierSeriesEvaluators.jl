# here we make use of the fact that for coefficients that satisfy A[i] = A[-i]' produce
# hermitian-valued Fourier series since z*A[i] + z'*A[-i] = (z*A[i]) + (z*A[i])'

@fastmath function hermitian_fourier_evaluate(C::AbstractVector{T}, x, k=inv(oneunit(x)), a=Val(0), shifts::Integer=0) where {T}

    u = k*x                 # unitless, could be complex
    s = size(C, 1); isev = iseven(s)
    c = firstindex(C, N)    # Find the index offset
    c += M = div(s, 2)      # translate to find axis center
    c0 = c+shift            # initial offset index
    z0 = cis(u*c0)          # initial offset phase

    ax = axes(C,1)

    # apply global Fourier multiplier z0*(im*k)^a (k^a could change type)
    z = _iszero(a)  ? z0            :
        _isone(a)   ? z0*im*k       :
        _istwo(a)   ? z0*(im*k)^2   : z0*_pow(im*k, a)

    # initial Fourier coefficient z*c0^a (won't change type)
    b = isev        ? zero(z)   :
        _iszero(a)  ? z         :
        _isone(a)   ? z*c0      :
        _istwo(a)   ? z*c0^2    : z*_pow(c0, a)

    if N == 1
        @inbounds r = b * C[c]  # unroll first loop iteration
    else
        r = b * fourier_evaluate(selectdim(C, N, c), x_, k_, a_, shift_)
    end
    s == 1 && return r      # fast return for singleton dimensions

    _dz, dz_ = cis_inv(step(ax)*u)   # forward-backward-winding phase steps

    z_, c_ = z, c                  # initial winding phase, index
    _z, _c = isev ? (z*dz_, c-1) : (z, c)

    for n in 1:M            # symmetric winding loop
        _z *= _dz           # forward--winding phase
        z_ *= dz_           # backward-winding phase

        _i = _c + n         # forward--winding index
        i_ = c_ - n         # backward-winding index

        # winding coefficients
        _b,b_ = _iszero(a)  ? (_z,              z_)                 :
                _isone(a)   ? (_z*(_i+shift),   z_*(i_+shift))      :
                _istwo(a)   ? (_z*(_i+shift)^2, z_*(i_+shift)^2)    : (_z*_pow(_i+shift, a), z_*_pow(i_+shift, a))

        if N == 1
            @inbounds r += _b * C[_i] + b_ * C[i_]
        else
            r += _b * fourier_evaluate(selectdim(C, N, _i), x_, k_, a_, shift_) +
                 b_ * fourier_evaluate(selectdim(C, N, i_), x_, k_, a_, shift_)
        end
    end
    return r
end
