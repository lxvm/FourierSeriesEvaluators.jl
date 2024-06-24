module FourierSeriesEvaluatorsStaticArraysExt
import FourierSeriesEvaluators: _hermitianpart, _mul
using StaticArrays: SHermitianCompact, StaticMatrix, SMatrix
using LinearAlgebra: Hermitian, checksquare

@inline @fastmath _mul(z::Complex, A::SHermitianCompact) = z * SMatrix(A)
_hermitianpart(A::StaticMatrix) = SHermitianCompact{checksquare(A)}(Hermitian(A, 'L'))
#=
# the code above runs as efficiently as this manually written version
function _hermitianpart(A::StaticMatrix)
    n = checksquare(A)
    M = ntuple(Val(n)) do j
        o = (j-1)*(n+1)
        ntuple(Val(n-j+1)) do i
            a = A[i+o]
            a isa Complex || return a # avoid type instability for real coefficients
            i == 1 ? complex(real(a)) : a
        end
    end
    SHermitianCompact(SVector(_tcat(M...)))
end
_tcat(a, b) = (a..., b...)
_tcat(a,b,c...) = _tcat(_tcat(a, b), c...)
=#
end
