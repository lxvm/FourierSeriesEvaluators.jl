
struct TranslatedFourierSeries{N,T,iip,F<:AbstractFourierSeries{N,T,iip}} <: AbstractFourierSeries{N,T,iip}
    f::F
    y::NTuple{N,T}
end


function allocate(ts::TranslatedFourierSeries, x, dim)
    f_cache = allocate(ts.f, x, dim)
    return f_cache
end

function contract!(cache, ts::TranslatedFourierSeries, x, dim::Val{d}) where {d}
    fx = contract!(cache, ts.f, x-ts.y[d], dim)
    return TranslatedFourierSeries(fx, deleteat_(ts.y, dim))
end

function evaluate!(cache, ts::TranslatedFourierSeries{1}, x)
    return evaluate!(cache, ts.f, x-only(ts.y))
end

period(ts::TranslatedFourierSeries) = period(ts.f)
frequency(ts::TranslatedFourierSeries) = frequency(ts.f)

show_details(ts::TranslatedFourierSeries) = " shifted by $(ts.y)"

