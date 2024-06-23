using Test
using LinearAlgebra

using StaticArrays

using FourierSeriesEvaluators
using FourierSeriesEvaluators: _check_coeffs_hermsym, _check_coeffs_centered
using FourierSeriesEvaluators: workspace_allocate, workspace_evaluate

@testset "hermitian checks" begin
    n = 3
    T = SMatrix{3,3,ComplexF64,9}
    c = rand(T, 2n+1,2n+1)
    @test !_check_coeffs_hermsym(c, (nothing, nothing))
    c = c + reverse(adjoint.(c))
    @test _check_coeffs_hermsym(c, (nothing, nothing))
    @test !_check_coeffs_centered(c, (-n, -n))
    @test _check_coeffs_centered(c, (-n-1, -n-1))
    f = FourierSeries(rand(T, 2n+1, 2n+1); period=1.0, offset=-n-1)
    @test_throws ArgumentError HermitianFourierSeries(f)
    f = FourierSeries(c; period=1.0, offset=-n)
    @test_throws ArgumentError HermitianFourierSeries(f)
    # even-numbered
    c = rand(SMatrix{3,3,ComplexF64,9}, 2n, 2n)
    c = c + reverse(adjoint.(c))
    f = FourierSeries(c; period=1.0, offset=-n-1)
    @test_throws ArgumentError HermitianFourierSeries(f)
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
        rand(Complex{T}, ntuple(_->2n+1,d)...),
        # [rand(Complex{T}, 3, 3) for _ in idx],
        rand(SMatrix{2,2,Complex{T},4}, ntuple(_->2n+1,d)...),
        rand(SMatrix{3,3,Complex{T},9}, ntuple(_->2n+1,d)...),
    ]
        c = a + reverse(adjoint.(a))
        f = FourierSeries(c; period=1.0, offset=-n-1)
        if first(c) isa AbstractMatrix && d == 1
            f2 = FourierSeries(UpperTriangular.(c); period=1.0, offset=-n-1)
        end
        hf = HermitianFourierSeries(f)
        for _ in 1:ntest
            x = rand(T, d)
            hfx = @inferred hf(x)
            @test ishermitian(hfx isa AbstractMatrix ? Matrix(hfx) : hfx)
            @test f(x) ≈ hfx
            if first(c) isa AbstractMatrix && d == 1
                @test hfx ≈ Hermitian(f2(x))
            end
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
        rand(Complex{T}, ntuple(_->2n+1,d)...),
        # [rand(Complex{T}, 3, 3) for _ in idx],
        rand(SMatrix{2,2,Complex{T},4}, ntuple(_->2n+1,d)...),
        rand(SMatrix{3,3,Complex{T},9}, ntuple(_->2n+1,d)...),
    ]
        c = Array{eltype(a)}(undef, ninner, 2n+1,2n+1)
        for i in axes(c,1)
            c[i,:,:] .= a[i,:,:] + reverse(adjoint.(a[i,:,:]))
        end
        f = FourierSeries(c, d; period=1.0, offset=-n-1)
        @test_throws ArgumentError hf = HermitianFourierSeries(f)
        #=
        for _ in 1:ntest
            x = rand(T, d)
            hfx = @inferred hf(x)
            @test all(ishermitian, hfx)
            @test f(x) ≈ hfx
        end
        =#
    end
end

@testset "derivative" begin
    order = 2
    d = 2
    n = 5
    ntest = 3
    T = Float64
    c = rand(SMatrix{2,2,Complex{T},4}, ntuple(_->2n+1,d)...)
    c = c + reverse(adjoint.(c))
    f = FourierSeries(c; period=1.0, offset=-n-1)
    hf = HermitianFourierSeries(f)
    df = DerivativeSeries{order}(f)
    dhf = DerivativeSeries{order}(hf)
    for i in 1:ntest
        x = rand(d)
        dfx = @inferred df(x)
        dhfx = @inferred dhf(x)
        for (n,(dfxi, dhfxi)) in enumerate(zip(dfx, dhfx))
            if n == (0+1)
                @test ishermitian(dhfxi)
                @test dfxi ≈ dhfxi
            elseif n == (1+1)
                for i in 1:d
                    @test ishermitian(dhfxi[i])
                    @test dfxi[i] ≈ dhfxi[i]
                end
            elseif n == (2+1)
                for j in 1:d
                    for i in 1:d-j+1
                        @test ishermitian(dhfxi[j][i])
                        @test dfxi[j][i] ≈ dhfxi[j][i]
                    end
                end
            elseif n == (3+1)
                for k in 1:d
                    for j in 1:d-j+1
                        for i in 1:d-j-k+2
                            @test ishermitian(dhfxi[k][j][i])
                            @test dfxi[k][j][i] ≈ dhfxi[k][j][i]
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
    T = SMatrix{2,2,ComplexF64,4}
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
        hfip = HermitianFourierSeries(fip)
        x = tuple(rand(nvar)...)
        for sip in (hfip, ManyFourierSeries(hfip,hfip,period=periods), JacobianSeries(hfip))
            wsip = workspace_allocate(sip, x)
            @test workspace_evaluate(wsip, x) == sip(x)
        end
    end
    =#
    C = C + reverse(adjoint.(C))
    periods = rand(d)
    f = FourierSeries(C; period=periods, offset=-n-1)
    hf = HermitianFourierSeries(f)
    x = tuple(rand(d)...)
    for s in (hf, ManyFourierSeries(hf,hf,period=periods), JacobianSeries(hf))
        ws = workspace_allocate(s, x)
        @test workspace_evaluate(ws, x) == s(x)
    end
end
