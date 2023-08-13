"""
    FourierWorkspace

A workspace for storing a Fourier series and the intermediate arrays used for evaluations.
Given a `ws::FourierWorkspace`, you can evaluate it at a point `x` with `ws(x)`.

All functionality is implemented by the [`workspace_allocate`](@ref),
[`workspace_contract!`](@ref), and [`workspace_evaluate!`](@ref) routines, which allow
multiple caches within each dimension of evaluation to enable parallel workloads.
"""
struct FourierWorkspace{S,C}
    series::S
    cache::C
end

"""
    workspace_allocate(s::AbstractFourierSeries{N}, x::NTuple{N}, [len::NTuple{N}=ntuple(one,N)])

Allocates a [`FourierWorkspace`](@ref) for the Fourier series `s` that can be used to
evaluate the series multiple times without allocating on-the-fly. The `len` argument can
indicate how many copies of workspace should be made for each variable for downstream use in
parallel workloads.

The workspace is constructed recursively starting from the outer dimension and moving
towards the inner dimension so as to access memory contiguously. Thus, the outer dimension
has `len[N]` workspace copies and each of these has `len[N-1]` workspaces for the next
variable. In total there are `prod(len)` leaf-level caches to use for parallel workloads.
"""
function workspace_allocate(s::AbstractFourierSeries{N}, x::NTuple{N,Any}, len::NTuple{N,Integer}=fill_ntuple(1,N)) where{N}
    # Only the top-level workspace has an AbstractFourierSeries in the series field
    # In the lower level workspaces the series field has a cache that can be contract!-ed
    # into a series
    dim = Val(N)
    ws = ntuple(Val(len[N])) do n
        cache = allocate(s, x[N], dim)
        if N == 1
            return cache
        else
            t = contract!(cache, s, x[N], dim)
            return FourierWorkspace(cache, workspace_allocate(t, x[1:N-1], len[1:N-1]).cache)
        end
    end
    return FourierWorkspace(s, ws)
end

"""
    workspace_contract!(ws, x, [i=1])

Returns a workspace with the series contracted at variable `x` in the outer dimension. The
index `i` selects which workspace in the cache to assign the new data.
"""
function workspace_contract!(ws, x, i=1)
    dim = Val(ndims(ws.series)) # we select the outer dimension so the inner are contiguous
    s = contract!(ws.cache[i].series, ws.series, x, dim)
    return FourierWorkspace(s, ws.cache[i].cache)
end

"""
    workspace_evaluate!(ws, x, [i=1])

Return the 1-d series evaluated at the variable `x`, using cache sector `i`.
"""
workspace_evaluate!(ws, x, i=1) = evaluate!(ws.cache[i], ws.series, x)

function workspace_evaluate(ws::FourierWorkspace{<:AbstractFourierSeries{1}}, (x,)::NTuple{1})
    return workspace_evaluate!(ws, x)
end
function workspace_evaluate(ws::FourierWorkspace{<:AbstractFourierSeries{N}}, x::NTuple{N,Any}) where {N}
        return workspace_evaluate(workspace_contract!(ws, x[N]), x[1:N-1])
end

function (ws::FourierWorkspace)(x)
    (N = ndims(ws.series)) == length(x) || throw(ArgumentError("number of input variables doesn't match those in series"))
    return workspace_evaluate(ws, NTuple{N}(x))
end
