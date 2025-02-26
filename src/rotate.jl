"""
    rotate(f::FourierSeries, A::AbstractMatrix{<:Integer})

Apply a rotation to a Fourier series `f(x)` and return a rotated series `g(x) =
f(A*x)`. The coordinate transformation `A` must be an integer matrix. The
transformation is applied to the series coefficients themselves by rewriting
``f(Ax) = \\sum_{k} f_k e^{ik\\cdot Ax} = \\sum_{k} f_k e^{iA^Tk\\cdot x} =
\\sum{k' = A^Tk} f_{k'} e^{ik'\\cdot x}``.
"""
function rotate(f::FourierSeries, A::AbstractMatrix{<:Integer})
    c = f.c
    o = CartesianIndex(f.o)
    t = f.t
    iAt = inv(transpose(A))
    @assert all(isequal(t[1]), t) "series periods must be commensurate"
    kmold = kmin = first(CartesianIndices(c))+o
    kxold = kmax = last(CartesianIndices(c))+o
    tnew = ntuple(n -> t[1]/gcd(ntuple(m -> A[m,n], Val(ndims(f)))...), Val(ndims(f)))
    for i in CartesianIndices(ntuple(_-> 0:1,Val(ndims(f))))
        v = kmold + CartesianIndex(ntuple(s -> i[s]*(kxold[s]-kmold[s]), Val(ndims(f))))
        u = A*[Tuple(v)...]
        for (n, un) in enumerate(Tuple(u))
            if un < kmin[n]
                kmin = CartesianIndex(ntuple(m -> m==n ? un : kmin[m], Val(ndims(f))))
            end
            if un > kmax[n]
                kmax = CartesianIndex(ntuple(m -> m==n ? un : kmax[m], Val(ndims(f))))
            end
        end
    end
    C = zeros(eltype(c), Tuple(kmax - kmin) .+ 1)
    onew = Tuple(kmin-one(kmin))
    for i in CartesianIndices(c)
        C[CartesianIndex(Tuple(transpose(A)*[Tuple(i+o)...]))-CartesianIndex(onew)] = c[i]
    end
    return FourierSeries(C; period=t, deriv=f.a, offset=onew)
end