using Test
using LinearAlgebra

using OffsetArrays: OffsetArray, Origin
using StaticArrays

using FourierSeriesEvaluators
using FourierSeriesEvaluators: raise_multiplier

# TODO: validate the reference functions against FFTW
include("fourier_reference.jl")

@testset "FourierSeriesEvaluators" begin

    @testset "fourier_kernel" begin
        nxtest = 5 # number of test points
        for nctest in 1:7 # number of coefficients to test
        # test number of array dimensions and different coefficient types
        @testset "ndims=$n coefficient_type=$T" for n in 1:3, T in (Float64, ComplexF64, SMatrix{2,2,Float64,4}, SMatrix{8,8,ComplexF64,64})
            C = rand(T, ntuple(i -> nctest, n)...)
            R = Array{T,n-1}(undef, ntuple(i -> nctest, n-1)...) # preallocate result array for kernel
            origin = div(nctest,2) # new Cartesian index for first element of array
            OC = OffsetArray(C, Origin(ntuple(i -> -origin, n)...))
            OR = OffsetArray(R, Origin(ntuple(i -> -origin, n-1)...))
            # real and complex inputs and different orders of derivative
            for x in (rand(nxtest)..., rand(ComplexF64, nxtest)...), a in (0, 1, 2), dim in 1:n
                period = 2.11; k = 2pi/inv(period)
                shift = 0
                ref = ref_contract(C, x, k, a, dim)
                oref = ref_contract(OC, x, k, a, dim)
                if !(eltype(T) <: Complex)
                    @test_throws ArgumentError fourier_contract!(R, C, x, k, a, shift, Val(dim))
                else
                    # compare to reference evaluators
                    # @show x a n
                    @test ref ≈ fourier_contract!(R, C, x, k, a, shift, Val(dim))
                    @test oref ≈ fourier_contract!(OR, OC, x, k, a, shift, Val(dim))
                    # test the shift
                    @test parent(OR) ≈ fourier_contract!(R, parent(OC), x, k, a, -(origin+1), Val(dim))
                end
                # test the allocating version
                @test ref ≈ fourier_contract(C, x, k, a, shift, Val(dim))
                @test oref ≈ fourier_contract(OC, x, k, a, shift, Val(dim))
                # test the 1D evaluators
                n == 1 || continue
                # compare to reference evaluators
                @test ref_evaluate(C, x, k, a) ≈ fourier_evaluate(C, x, k, a)
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(OC, x, k, a)
                # test the shift
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(parent(OC), x, k, a, -(origin+1))
            end
        end
        end
    end

    @testset "FourierSeries" begin
        d = 3; nxtest=5
        n = 11; m = div(n,2)
        for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
            C = rand(T, ntuple(_->n, d)...)
            OC = OffsetArray(C, ntuple(_->-m:m, d)...)
            for _ in 1:nxtest
                x = rand(d)
                # test period
                periods = rand(d)
                f = FourierSeries(C, period=periods)
                @test all(f.p .≈ periods)
                @test ndims(f) == d
                @test f(x) ≈ ref_evaluate(C, x, 2pi ./ periods)
                # test derivative
                for a in (0, 1, 2)
                    f = FourierSeries(C, period=1, deriv=a)
                    @test f(x) ≈ ref_evaluate(C, x, 2pi, a)
                end
                # test offset
                f = FourierSeries(OC, period=1)
                @test f(x) ≈ ref_evaluate(OC, x, 2pi)
                f = FourierSeries(C, period=1, offset=-m-1)
                @test f(x) ≈ ref_evaluate(OC, x, 2pi)
                # test shift
                q = rand(d)
                f = FourierSeries(C, period=1, shift=q)
                @test f(x) ≈ ref_evaluate(C, x-q, 2pi)
            end
        end
    end

    @testset "ManyFourierSeries" begin
        d = 3; nxtest=5
        n = 11; m = div(n,2)
        nfs = 10
        for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
            periods = rand(d)
            fs = ntuple(_ -> FourierSeries(rand(T, ntuple(_->n, d)...), period=periods), nfs)
            mfs = ManyFourierSeries(fs..., period=periods)
            # test period
            @test ndims(mfs) == d
            for _ in 1:nxtest
                x = rand(d)
                # test return value
                @test all(mfs(x) .≈ map(f -> f(x), fs))
            end
        end
    end

    @testset "GradientSeries" begin
        d = 3
        n = 11; m = div(n,2)
        for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
            periods = rand(d)
            f = FourierSeries(rand(T, ntuple(_->n, d)...), period=periods)
            g = GradientSeries(f)
            gs = map(dim -> FourierSeries(f.c, f.p, f.k, raise_multiplier(f.a, Val(dim)), f.o, f.q), 1:d)
            for x in ((0,0,0), (0.1im,complex(0.2), complex(-0.3,0.4)), rand(d))
                fx, dfx = g(x)
                @test fx ≈ f(x)
                for i in 1:d
                    @test dfx[i] ≈ gs[i](x)
                end
            end
        end
    end

    @testset "HessianSeries" begin
        d = 3
        n = 11; m = div(n,2)
        for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
            periods = rand(d)
            f = FourierSeries(rand(T, ntuple(_->n, d)...), period=periods)
            h = HessianSeries(f)
            hs = map(1:d) do dim1
                map(dim1:d) do dim2
                    a = raise_multiplier(raise_multiplier(f.a, Val(dim1)), Val(dim2))
                    return FourierSeries(f.c, f.p, f.k, a, f.o, f.q)
                end
            end

            for x in ((0,0,0), (0.1im,complex(0.2), complex(-0.3,0.4)), rand(d))
                fx, dfx, d2fx = h(x)
                @test fx ≈ f(x)
                for i in 1:d
                    for j in ((i:d) .- (i-1))
                        @test d2fx[i][j] ≈ hs[i][j](x)
                    end
                end
            end
        end
    end
end
