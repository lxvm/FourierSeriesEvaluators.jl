using BenchmarkTools
using StaticArrays

using FourierSeriesEvaluators: fourier_contract!, fourier_contract, fourier_evaluate

suite = BenchmarkGroup()

suite["fourier_contract!"] = BenchmarkGroup(["kernel", "inplace"])
suite["fourier_contract"]  = BenchmarkGroup(["kernel", "allocating"])
suite["fourier_evaluate"]  = BenchmarkGroup(["kernel", "non-allocating"])

for M=[11, 12, 111, 112, 1111, 1112] # numbers of Fourier coefficients to benchmark
    for T in (ComplexF64, SHermitianCompact{3,ComplexF64,6}, SMatrix{3,3,ComplexF64,9})
        R = Array{T,0}(undef)
        C = rand(T, M)
        for x in (0.6914384774549351, 0.0393911707729353 + 0.5777059044321278im), a in (0, 1, 2, 2.1)
            k = oftype(x, 1)
            suite["fourier_contract!"][M, T, typeof(x), a] = @benchmarkable fourier_contract!($R, $C, $x, $k, $a)
            suite["fourier_contract"][M, T, typeof(x), a]  = @benchmarkable fourier_contract($C, $x, $k, $a)
            suite["fourier_evaluate"][M, T, typeof(x), a]  = @benchmarkable fourier_evaluate($C, $x, $k, $a)
        end
    end
end

paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(suite, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(suite)
    BenchmarkTools.save(paramspath, params(suite));
end

# results = run(suite, verbose = true, seconds = 1)
