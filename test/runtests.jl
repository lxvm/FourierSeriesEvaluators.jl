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
                    @test_throws InexactError fourier_contract!(R, C, x, k, a, shift, Val(dim))
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
                @test ref_evaluate(C, x, k, a) ≈ fourier_evaluate(C, (x,), (k,), (a,))
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(OC, (x,), (k,), (a,))
                # test the shift
                @test ref_evaluate(OC, x, k, a) ≈ fourier_evaluate(parent(OC), (x,), (k,), (a,), (-(origin+1),))
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

    @testset "DerivativeSeries" for order in 1:3
        d = 3
        n = 11; m = div(n,2)
        for T in (ComplexF64, SMatrix{5,5,ComplexF64,25})
            periods = rand(d)
            f = FourierSeries(rand(T, ntuple(_->n, d)...), period=periods)
            ds = DerivativeSeries{order}(f)
            dOs = if order == 1
                map(dim -> FourierSeries(f.c, f.p, f.f, raise_multiplier(f.a, Val(dim)), f.o, f.q), 1:d)
            elseif order == 2
                map(1:d) do dim1
                    map(dim1:d) do dim2
                        a = raise_multiplier(raise_multiplier(f.a, Val(dim1)), Val(dim2))
                        return FourierSeries(f.c, f.p, f.f, a, f.o, f.q)
                    end
                end
            elseif order == 3
                map(1:d) do dim1
                    map(dim1:d) do dim2
                        map(dim2:d) do dim3
                            a = raise_multiplier(raise_multiplier(raise_multiplier(f.a, Val(dim1)), Val(dim2)), Val(dim3))
                            return FourierSeries(f.c, f.p, f.f, a, f.o, f.q)
                        end
                    end
                end
            else
                throw(ArgumentError("no test case for order $order derivatives"))
            end
            for x in ((0,0,0), (0.1im,complex(0.2), complex(-0.3,0.4)), rand(d))
                fx, dOfx = ds(x)[[1,order+1]]
                @test fx ≈ f(x)
                if order == 1
                    for i in 1:d
                        @test dOfx[i] ≈ dOs[i](x)
                    end
                elseif order == 2
                    for i in 1:d
                        for j in i:d
                            @test dOfx[i][j-i+1] ≈ dOs[i][j-i+1](x)
                        end
                    end
                elseif order == 3
                    for i in 1:d
                        for j in i:d
                            for k in j:d
                                @test dOfx[i][j-i+1][k-j+1] ≈ dOs[i][j-i+1][k-j+1](x)
                            end
                        end
                    end
                else
                    throw(ArgumentError("no test case for order $order derivatives"))
                end
            end
        end
    end

    @testset "inplace" begin
        d = 3
        n = 11
        T = ComplexF64
        for nvar in 1:d
            periods = rand(nvar)
            C = rand(T, ntuple(_->n, d)...)
            sizes = ntuple(_ -> n, d-nvar)
            s = FourierSeries(SArray{Tuple{sizes...},T,d-nvar,prod(sizes)}[view(C,ntuple(_ -> (:), d-nvar)..., i) for i in CartesianIndices(axes(C)[end-nvar+1:end])], period=periods)
            sip = FourierSeries(C, nvar, period=periods)
            x = rand(nvar)
            @test (@inferred sip(x)) ≈ s(x)
            ms = ManyFourierSeries(s, s)
            msip = ManyFourierSeries(sip, sip)
            @test all((@inferred msip(x)) .≈ ms(x))
            ds = @inferred HessianSeries(s)(x)
            dsip = @inferred HessianSeries(sip)(x)
            @test dsip[1] ≈ ds[1]
            @test all(dsip[2] .≈ ds[2])
            @test all(map(i -> all(dsip[3][i] .≈ ds[3][i]), 1:nvar))
        end
    end

    # @testset "workspace" begin

    # end
end
