using Test
using LinearAlgebra

using OffsetArrays: OffsetArray, Origin
using StaticArrays

using FourierSeriesEvaluators: fourier_contract!, fourier_contract, fourier_evaluate

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
            for x in (rand(nxtest)..., rand(ComplexF64, nxtest)...), (a,va) in ((0,Val(0)), (1,Val(1)), (2,2))
                period = 2.11; k = 2pi/inv(period)
                ref = ref_contract(C, x, k, a)
                oref = ref_contract(OC, x, k, a)
                if !(eltype(T) <: Complex)
                    @test_throws ArgumentError fourier_contract!(R, C, x, k, va)
                else
                    # compare to reference evaluators
                    @test ref ≈ fourier_contract!(R, C, x, k, va)
                    @test oref ≈ fourier_contract!(OR, OC, x, k, va)
                    # test the shift
                    @test parent(OR) ≈ fourier_contract!(R, parent(OC), x, k, va, -(origin+1))
                end
                # test the allocating version
                @test ref ≈ fourier_contract(C, x, k, va)
                @test oref ≈ fourier_contract(OC, x, k, va)
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
    #=
    @testset "FourierSeries3D" begin
        
    end

    @testset "FourierSeries" begin
        
    end

    @testset "FourierSeriesDerivative" begin
        
    end

    @testset "OffsetFourierSeries" begin
        
    end

    @testset "ManyFourierSeries" begin
        
    end

    @testset "ManyOffsetsFourierSeries" begin
        
    end
    =#
end