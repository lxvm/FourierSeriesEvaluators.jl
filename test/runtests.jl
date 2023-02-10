using Test
using LinearAlgebra

using OffsetArrays: OffsetArray, Origin
using StaticArrays

using FourierSeriesEvaluators

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
            for x in (rand(nxtest)..., rand(ComplexF64, nxtest)...), (a,va) in ((0,Val(0)), (1,Val(1)), (2,2)), dim in 1:n
                period = 2.11; k = 2pi/inv(period)
                shift = 0
                ref = ref_contract(C, x, k, a, dim)
                oref = ref_contract(OC, x, k, a, dim)
                if !(eltype(T) <: Complex)
                    @test_throws ArgumentError fourier_contract!(R, C, x, k, va, shift, Val(dim))
                else
                    # compare to reference evaluators
                    @test ref ≈ fourier_contract!(R, C, x, k, va, shift, Val(dim))
                    @test oref ≈ fourier_contract!(OR, OC, x, k, va, shift, Val(dim))
                    # test the shift
                    @test parent(OR) ≈ fourier_contract!(R, parent(OC), x, k, va, -(origin+1), Val(dim))
                end
                # test the allocating version
                @test ref ≈ fourier_contract(C, x, k, va, shift, Val(dim))
                @test oref ≈ fourier_contract(OC, x, k, va, shift, Val(dim))
                # test the 1D evaluators
                n == 1 || continue
                # compare to reference evaluators
                @test ref_evaluate(C, x, k, a) ≈ fourier_evaluate(C, x, k, va)
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(OC, x, k, va)
                # test the shift
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(parent(OC), x, k, va, -(origin+1))
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
                period = rand(d)
                f = FourierSeries(C, period=period)
                @test f(x) ≈ ref_evaluate(C, x, 2pi ./ period)
                # test derivative
                for (deriv, a) in ((Val(0), 0), (Val(1), 1), fill(rand(1:4, d), 2))
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
            mfs = ManyFourierSeries(fs...)
            # test period
            @test all(period(mfs) .≈ periods)
            for _ in 1:nxtest
                x = rand(d)
                # test return value
                @test all(mfs(x) .≈ map(f -> f(x), fs))
            end
        end
    end
    
    #=
    @testset "FourierSeries3D" begin
        
    end


    @testset "InplaceFourierSeries" begin
        
    end
    =#
end