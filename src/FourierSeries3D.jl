# TODO: allow an arbitrary offset
"""
    FourierSeries3D(f::Array{T,3}, [period=(1.0, 1.0, 1.0)])

This type is an `AbstractInplaceFourierSeries{3}` and unlike `FourierSeries` is
specialized for 3D Fourier series and does not allocate a new array every time
`contract` is called on it. This type stores the intermediate arrays used in a
calculation and assumes that the size of `f` on each axis is odd because it
treats the zeroth harmonic as the center of the array (i.e. `(size(f) .÷ 2) .+
1`).
"""
struct FourierSeries3D{T,a1,a2,a3} <: AbstractInplaceFourierSeries{3,T}
    period::NTuple{3,Float64}
    # fi where i represents array dims
    f3::Array{T,3}
    f2::Array{T,2}
    f1::Array{T,1}
    f0::Array{T,0}
end

function FourierSeries3D(f3::Array{T,3}, period=(1.0,1.0,1.0), orders=(0,0,0)) where T
    @assert all(map(isodd, size(f3)))
    # TODO convert T to a stable fourier_type
    f2 = Array{T,2}(undef, size(f3,1), size(f3, 2))
    f1 = Array{T,1}(undef, size(f3,1))
    f0 = Array{T,0}(undef)
    FourierSeries3D{T,orders...}(period, f3, f2, f1, f0)
end

period(f::FourierSeries3D) = f.period
value(f::FourierSeries3D) = only(f.f0)
@generated function contract!(f::FourierSeries3D{T,a1,a2,a3}, x, ::Val{d}) where {d,T,a1,a2,a3}
    quote
        fourier_kernel!($(Symbol(:(f.f), d-1)), f.$(Symbol(:f, d)), x, inv(f.period[$d]), Val($(Symbol(:a, d))))
        return f
    end
end
contract!(f::FourierSeries3D, x, ::Val{1}) = (f.f0[] = f(x); return f)

(f::FourierSeries3D{T,a1})(x) where {T,a1} =
    fourier_kernel(f.f1, x, inv(f.period[1]), Val{a1}())