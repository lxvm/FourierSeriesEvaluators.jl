module FourierSeriesEvaluatorsStaticArraysExt
import FourierSeriesEvaluators: _hermitianpart
using StaticArrays: SHermitianCompact, StaticMatrix

_hermitianpart(A::StaticMatrix) = SHermitianCompact(A)

end
