using Test
using LinearAlgebra

using StaticArrays

using FourierSeriesEvaluators
using FourierSeriesEvaluators: _check_coeffs_realsym, _check_coeffs_centered
using FourierSeriesEvaluators: workspace_allocate, workspace_evaluate

@testset "real symmetric checks" begin
    n = 3
    T = SHermitianCompact{3,Float64,6}
    c = rand(T, 2n+1,2n+1)
    @test !_check_coeffs_realsym(c, (nothing, nothing))
    c = c + reverse(c; dims=1)
    c = c + reverse(c; dims=2)
    @test _check_coeffs_realsym(c, (nothing, nothing))
    @test !_check_coeffs_centered(c, (-n, -n))
    @test _check_coeffs_centered(c, (-n-1, -n-1))
    f = FourierSeries(rand(T, 2n+1, 2n+1); period=1.0, offset=-n-1)
    @test_throws ArgumentError RealSymmetricFourierSeries(f)
    f = FourierSeries(c; period=1.0, offset=-n)
    @test_throws ArgumentError RealSymmetricFourierSeries(f)
    # even-numbered
    c = rand(SHermitianCompact{3,Float64,6}, 2n, 2n)
    c = c + reverse(c; dims=1)
    c = c + reverse(c; dims=2)
    f = FourierSeries(c; period=1.0, offset=-n-1)
    @test_throws ArgumentError RealSymmetricFourierSeries(f)
    #=
    ninner=4
    cc = Array{ComplexF64}(undef, ninner, 2n+1,2n+1)
    @test !_check_coeffs_hermsym(cc, (nothing, nothing))
    for i in axes(cc,1)
        h = rand(ComplexF64, 2n+1, 2n+1)
        cc[i,:,:] .= h + reverse(adjoint.(h))
    end
    @test _check_coeffs_hermsym(cc, (nothing, nothing))
    @test !_check_coeffs_centered(cc, (-n, -n))
    @test _check_coeffs_centered(cc, (-n-1, -n-1))
    =#
end

@testset "evaluate" for d in 1:2
    n = 5
    ntest = 10
    T = Float64
    # idx = Iterators.product(repeat([1:(2n+1)], d)...)
    for a in [
        rand(T, ntuple(_->2n+1,d)...),
        # [Symmetric(rand(T, 3, 3)) for _ in idx],
        rand(SHermitianCompact{2,T,3}, ntuple(_->2n+1,d)...),
        rand(SHermitianCompact{3,T,6}, ntuple(_->2n+1,d)...),
    ]
        c = a
        for dims in 1:d
            c = c + reverse(c; dims)
        end
        f = FourierSeries(c; period=1.0, offset=-n-1)
        rsf = RealSymmetricFourierSeries(f)
        for _ in 1:ntest
            x = rand(T, d)
            rsfx = @inferred rsf(x)
            @test issymmetric(rsfx isa AbstractMatrix ? Matrix(rsfx) : rsfx)
            @test isreal(rsfx)
            @test f(x) ≈ rsfx
        end
    end
end

@testset "inplace" for d in 1:2
    n = 5
    ntest = 10
    ninner = 4
    T = Float64
    # idx = Iterators.product(ninner, repeat([1:(2n+1)], d)...)
    for a in [
        rand(T, ntuple(_->2n+1,d)...),
        # [rand(Complex{T}, 3, 3) for _ in idx],
        rand(SHermitianCompact{2,T,3}, ntuple(_->2n+1,d)...),
        rand(SHermitianCompact{3,T,6}, ntuple(_->2n+1,d)...),
    ]
        c = a
        for i in axes(c,1)
            for dims in 1:d
                c[i,:,:] .+= reverse(c[i,:,:]; dims)
            end
        end
        f = FourierSeries(c, 1; period=1.0, offset=-n-1)
        @test_throws ArgumentError RealSymmetricFourierSeries(f)
    end
end

@testset "derivative" begin
    order = 2
    d = 2
    n = 5
    ntest = 3
    T = Float64
    c = rand(SHermitianCompact{2,T,3}, ntuple(_->2n+1,d)...)
    for dims in 1:d
        c .+= reverse(c; dims)
    end
    f = FourierSeries(c; period=1.0, offset=-n-1)
    rsf = RealSymmetricFourierSeries(f)
    df = DerivativeSeries{order}(f)
    drsf = DerivativeSeries{order}(rsf)
    for i in 1:ntest
        x = rand(d)
        dfx = @inferred df(x)
        drsfx = @inferred drsf(x)
        for (n,(dfxi, drsfxi)) in enumerate(zip(dfx, drsfx))
            if n == (0+1)
                @test isreal(drsfxi)
                @test issymmetric(Matrix(drsfxi))
                @test dfxi ≈ drsfxi
            elseif n == (1+1)
                for i in 1:d
                    @test isreal(drsfxi[i])
                    @test issymmetric(Matrix(drsfxi[i]))
                    @test dfxi[i] ≈ drsfxi[i]
                end
            elseif n == (2+1)
                for j in 1:d
                    for i in 1:d-j+1
                        @test isreal(drsfxi[j][i])
                        @test issymmetric(Matrix(drsfxi[j][i]))
                        @test dfxi[j][i] ≈ drsfxi[j][i]
                    end
                end
            elseif n == (3+1)
                for k in 1:d
                    for j in 1:d-j+1
                        for i in 1:d-j-k+2
                            @test isreal(drsfxi[k][j][i])
                            @test issymmetric(Matrix(drsfxi[k][j][i]))
                            @test dfxi[k][j][i] ≈ drsfxi[k][j][i]
                        end
                    end
                end
            else
                error("test not implemented for order")
            end
        end
    end
end

@testset "workspace" begin
    d = 3
    n = 3
    T = SHermitianCompact{2,Float64,3}
    C = rand(T, ntuple(_ -> 2n+1, d)...)
    #=
    # inplace
    for nvar in 1:d-1
        periods = rand(nvar)
        CC = Array{eltype(C)}(undef, size(C)...)
        id = CartesianIndices(axes(C)[d-nvar+1:end])
        for i in CartesianIndices(axes(C)[1:d-nvar])
            CC[i,id] .= C[i,id] + reverse(adjoint.(C[i,id]))
        end
        fip = FourierSeries(CC, nvar, period=periods)
        rsfip = RealSymmetricFourierSeries(fip)
        x = tuple(rand(nvar)...)
        for sip in (rsfip, ManyFourierSeries(rsfip,rsfip,period=periods), JacobianSeries(rsfip))
            wsip = workspace_allocate(sip, x)
            @test workspace_evaluate(wsip, x) == sip(x)
        end
    end
    =#
    for dims in 1:d
        C .+= reverse(C; dims)
    end
    periods = rand(d)
    f = FourierSeries(C; period=periods, offset=-n-1)
    rsf = RealSymmetricFourierSeries(f)
    x = tuple(rand(d)...)
    for s in (rsf, ManyFourierSeries(rsf,rsf,period=periods), JacobianSeries(rsf))
        ws = workspace_allocate(s, x)
        @test workspace_evaluate(ws, x) == s(x)
    end
end
