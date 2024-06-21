using Test
using LinearAlgebra

using StaticArrays

using FourierSeriesEvaluators

@testset "" begin
    n = 5
    ntest = 10
    T = Float64
    for a in [[rand(Complex{T}, 2, 2) for _ in 1:(2n+1)], [rand(Complex{T}, 3, 3) for _ in 1:(2n+1)]]
        c = a + reverse(adjoint.(a))
        f = FourierSeries(c; period=1.0, offset=-n-1)
        f2 = FourierSeries(UpperTriangular.(c); period=1.0, offset=-n-1)
        hf = HermitianFourierSeries(c[n+1:end]; period=1.0)
        for _ in 1:ntest
            x = rand(T)
            @test f(x) ≈ hf(x) ≈ Hermitian(f2(x))
        end
    end

    for a in [[rand(SMatrix{2,2,Complex{T},4}) for _ in 1:(2n+1), _ in 1:(2n+1)], [rand(SMatrix{3,3,Complex{T},9}) for _ in 1:(2n+1), _ in 1:(2n+1)]]
        c = a + reverse(adjoint.(a))
        f = FourierSeries(c; period=1.0, offset=-n-1)
        # f2 = FourierSeries(UpperTriangular.(c); period=1.0, offset=-n-1)
        hf = HermitianFourierSeries(c[n+1:end,:]; period=1.0)
        for _ in 1:ntest
            x = rand(T, 2)
            @test f(x) ≈ hf(x) #≈ Hermitian(f2(x))
        end
    end


end
