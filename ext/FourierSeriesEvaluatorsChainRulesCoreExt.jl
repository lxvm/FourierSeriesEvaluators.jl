module FourierSeriesEvaluatorsChainRulesCoreExt

using LinearAlgebra: tr
using FourierSeriesEvaluators: AbstractFourierSeries, JacobianSeries
import ChainRulesCore: rrule, frule
using ChainRulesCore: ProjectTo, NoTangent, @not_implemented, AbstractZero

# we can't make guarantees that this works for custom series, since outputs may not be arrays
function rrule(s::AbstractFourierSeries{N}, x::NTuple{N,Any}) where {N}
    project_x = ProjectTo(x)
    y, J = JacobianSeries(s)(x)
    # since we don't know the output type y (i.e. scalar, vector, matrix), we try the
    # Frobenius inner product for the projection of ∂y onto the tangent space
    fourier_pullback(∂y) = @not_implemented("no rrule for changes in Fourier coefficients"), project_x(map(z -> tr(z*∂y), J))
    y, fourier_pullback
end

function frule((Δself, Δx), s::AbstractFourierSeries{N}, x::NTuple{N,Any}) where {N}
    y, J = JacobianSeries(s)(x)
    if Δself isa AbstractZero
        return y, mapreduce(*, +, Δx, J)
    else
        return y, @not_implemented("no frule for changes in Fourier coefficients")
    end
end

# TODO: add rules for contract! and evaluate!, and minimize allocations if possible

end
