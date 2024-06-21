module FourierSeriesEvaluatorsStaticArraysExt
import FourierSeriesEvaluators: _hermitianpart, _add
using StaticArrays: SHermitianCompact, StaticMatrix

_hermitianpart(A::StaticMatrix) = SHermitianCompact(A)
@inline _add(x::T, y::T) where {T<:SHermitianCompact} = x + y # faster without fastmath

end
