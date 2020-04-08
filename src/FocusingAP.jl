module FocusingAP

using Distances
using NearestNeighbors
using StatsBase

using Random

using Printf
using LinearAlgebra
using SparseArrays
using Statistics

import Base: show
import StatsBase: IntegerVector, RealVector, RealMatrix, counts

using Clustering

"""
    AffinityPropResult <: ClusteringResult
The output of affinity propagation clustering ([`affinityprop`](@ref)).
# Fields
 * `exemplars::Vector{Int}`: indices of *exemplars* (cluster centers)
 * `assignments::Vector{Int}`: cluster assignments for each data point
 * `iterations::Int`: number of iterations executed
 * `converged::Bool`: converged or not
 * `energy::Flaot64`: computes configuration energy
"""
mutable struct AffinityPropResult <: ClusteringResult
    exemplars::Vector{Int}      # indexes of exemplars (centers)
    assignments::Vector{Int}    # assignments for each point
    counts::Vector{Int}         # number of data points in each cluster
    iterations::Int             # number of iterations executed
    converged::Bool             # converged or not
    energy::Float64             # energy of the configuration
end

# const _afp_default_maxiter = 200
# const _afp_default_damp = 0.5
# const _afp_default_y = 10.0
# const _afp_default_γ = 0.01
# const _afp_default_γfact = 1.0
# const _afp_default_tol = 1.0e-6
# const _afp_default_display = :none
#
# const DisplayLevels = Dict(:none => 0, :final => 1, :iter => 2)
#
# display_level(s::Symbol) = get(DisplayLevels, s) do
#     throw(ArgumentError("Invalid value for the 'display' option: $s."))
# end

# function affinitypropR(S::DenseMatrix{T};
#                       maxiter::Integer=_afp_default_maxiter,
#                       tol::Real=_afp_default_tol,
#                       damp::Real=_afp_default_damp,
#                       y::Float64=_afp_default_y,
#                       γ::Float64=_afp_default_γ,
#                       γfact::Float64=_afp_default_γfact,
#                       display::Symbol=_afp_default_display,
#                       run_vanilla::Bool=false,
#                       zero_init::Bool=false) where T<:AbstractFloat
#
#     n = size(S, 1)
#     size(S, 2) == n || throw(ArgumentError("S must be a square matrix ($(size(S)) given)."))
#     n >= 2 || throw(ArgumentError("At least two data points are required ($n given)."))
#     tol > 0 || throw(ArgumentError("tol must be a positive value ($tol given)."))
#     0 <= damp < 1 || throw(ArgumentError("damp must be a non-negative real value below 1 ($damp given)."))
#
#     _affinityprop(S, round(Int, maxiter), tol, convert(T, damp), display_level(display), y, γ, γfact, run_vanilla, zero_init)
# end

#function _affinityprop(S::DenseMatrix{T}, o::Dict{Symbol,Any}) where T<:AbstractFloat
function _affinityprop(S::DenseMatrix{T};
                       maxiter::Int=1000,
                       tol::T=1e-6,
                       init::String="unif",
                       α::T=0.1,
                       damp::T=0.5,
                       print::Bool=true,
                       print_res::Bool=false) where T<:AbstractFloat
    n = size(S, 1)
    n2 = n * n

    if init == "gauss"
        init_fun = randn
    elseif init == "unif"
        init_fun = rand
    else
        error("Wrong init. Got $init")
    end

    R = α .* init_fun(T, n, n)
    A = α .* init_fun(T, n, n)

    # I dont want this
    # TODO; use multiple dispatch
    A_down = zeros(T, n, n)

    # prepare storages
    Rt = Matrix{T}(undef, n, n)
    At = Matrix{T}(undef, n, n)

    if print
        @printf "%7s %12s | %8s \n" "Iters" "objv-change" "exemplars"
        println("-----------------------------------------------------")
    end

    t = 0
    converged = false
    while !converged && t < maxiter

        _afp_compute_r!(Rt, S, A, A_down)
        _afp_dampen_update!(R, Rt, damp)

        _afp_compute_a!(At, R)
        _afp_dampen_update!(A, At, damp)

        # determine convergence
        ch = max(Linfdist(A, At), Linfdist(R, Rt)) / (one(T) - damp)
        converged = (ch < tol)

        t += 1

        if print
            # count the number of exemplars
            ne = _afp_count_exemplars(A, R)
            @printf("%7d %12.4e | %8d\n", t, ch, ne)
        end
    end

    # extract exemplars and assignments
    exemplars = _afp_extract_exemplars(A, R)
    assignments, counts = _afp_get_assignments(S, exemplars)
    energy = _afp_compute_energy(S, exemplars, assignments)

    if print || print_res
        if converged
            println("AP converged with $t iterations: $(length(exemplars)) exemplars.")
        else
            println("AP terminated without convergence after $t iterations: $(length(exemplars)) exemplars.")
        end
    end

    # produce output struct
    return AffinityPropResult(exemplars, assignments, counts, t, converged, energy)
end

#function _affinitypropR(S::DenseMatrix{T}, o::Dict{Symbol,Any}) where T<:AbstractFloat
function _affinitypropR(S::DenseMatrix{T};
                        maxiter::Int=1000,
                        tol::T=1e-6,
                        init::String="unif",
                        α::T=0.1,
                        damp::T=0.5,
                        γ::T=0.1,
                        γfact::T=0.001,
                        y::T=1.0,
                        yfact::T=0.0,
                        print::Bool=true,
                        print_res::Bool=false) where T<:AbstractFloat
    n = size(S, 1)
    n2 = n * n

    if init == "gauss"
        init_fun = randn
    elseif init == "unif"
        init_fun = rand
    else
        error("Wrong init. Got $init")
    end

    R = α .* init_fun(T, n, n)
    A = α .* init_fun(T, n, n)

    R_ref = α .* init_fun(T, n, n)
    A_ref = α .* init_fun(T, n, n)

    A_up = α .* init_fun(T, n, n)  # from replica to reference
    A_down = α .* init_fun(T, n, n)  # from reference to replica

    # prepare storages
    Rt = Matrix{T}(undef, n, n)
    At = Matrix{T}(undef, n, n)

    Rt_ref = Matrix{T}(undef, n, n)
    At_ref = Matrix{T}(undef, n, n)

    At_up = Matrix{T}(undef, n, n)
    At_down = Matrix{T}(undef, n, n)

    if print
        @printf "%7s %12s | %8s \n" "Iters" "objv-change" "exemplars"
        println("-----------------------------------------------------")
    end

    t = 0
    converged = false
    while !converged && t < maxiter

        _afp_compute_r!(Rt, S, A, A_down)
        _afp_dampen_update!(R, Rt, damp)

        _afp_compute_a!(At, R)
        _afp_dampen_update!(A, At, damp)

        _afp_compute_r_ref!(Rt_ref, A_ref, A_up, y)
        _afp_dampen_update!(R_ref, Rt_ref, damp)

        _afp_compute_a!(At_ref, R_ref)
        _afp_dampen_update!(A_ref, At_ref, damp)

        _afp_compute_a_down!(At_down, A_ref, A_up, γ, y)
        _afp_dampen_update!(A_down, At_down, damp)

        _afp_compute_a_up!(At_up, A, S, γ)
        _afp_dampen_update!(A_up, At_up, damp)

        # determine convergence
        ch = max(Linfdist(A, At), Linfdist(R, Rt)) / (one(T) - damp)
        converged = (ch < tol)

        t += 1
        γ *= (1.0 + γfact)
        y *= (1.0 + yfact)

        if print
            # count the number of exemplars
            ne = _afp_count_exemplars(A, R)
            @printf("%7d %12.4e | %8d\n", t, ch, ne)
        end
    end

    # extract exemplars and assignments
    exemplars = _afp_extract_exemplars(A, R)
    assignments, counts = _afp_get_assignments(S, exemplars)
    energy = _afp_compute_energy(S, exemplars, assignments)

    if print || print_res
        if converged
            println("R-AP converged with $t iterations: $(length(exemplars)) exemplars.")
        else
            println("R-AP terminated without convergence after $t iterations: $(length(exemplars)) exemplars.")
        end
    end

    # produce output struct
    return AffinityPropResult(exemplars, assignments, counts, t, converged, energy)
end

# compute responsibilities
function _afp_compute_r!(R::Matrix{T}, S::DenseMatrix{T}, A::Matrix{T}, A_down::Matrix{T}) where T
    n = size(S, 1)

    I1 = Vector{Int}(undef, n)  # I1[i] is the column index of the maximum element in (A+S)[i,:]
    Y1 = Vector{T}(undef, n)    # Y1[i] is the maximum element in (A+S)[i,:]
    Y2 = Vector{T}(undef, n)    # Y2[i] is the second maximum element in (A+S)[i,:]

    # Find the first and second maximum elements along each row
    @inbounds for i = 1:n
        v1 = A[i,1] + S[i,1] + A_down[i,1]
        v2 = A[i,2] + S[i,2] + A_down[i,2]
        if v1 > v2
            I1[i] = 1
            Y1[i] = v1
            Y2[i] = v2
        else
            I1[i] = 2
            Y1[i] = v2
            Y2[i] = v1
        end
    end
    @inbounds for j = 3:n, i = 1:n
        v = A[i,j] + S[i,j] + A_down[i,j]
        if v > Y2[i]
            if v > Y1[i]
                Y2[i] = Y1[i]
                I1[i] = j
                Y1[i] = v
            else
                Y2[i] = v
            end
        end
    end

    # compute R values
    @inbounds for j = 1:n, i = 1:n
        mv = (j == I1[i] ? Y2[i] : Y1[i])
        R[i,j] = S[i,j] + A_down[i,j] - mv
    end

    return R
end

# compute auxiliary replica responsibilities
function _afp_compute_r_ref!(R::Matrix{T}, A::Matrix{T}, A_up::Matrix{T}, y::T) where T
    n = size(A, 1)

    I1 = Vector{Int}(undef, n)  # I1[i] is the column index of the maximum element in (A+S)[i,:]
    Y1 = Vector{T}(undef, n)    # Y1[i] is the maximum element in (A+S)[i,:]
    Y2 = Vector{T}(undef, n)    # Y2[i] is the second maximum element in (A+S)[i,:]

    # Find the first and second maximum elements along each row
    @inbounds for i = 1:n
        v1 = A[i,1] + y * A_up[i,1]
        v2 = A[i,2] + y * A_up[i,2]
        if v1 > v2
            I1[i] = 1
            Y1[i] = v1
            Y2[i] = v2
        else
            I1[i] = 2
            Y1[i] = v2
            Y2[i] = v1
        end
    end
    @inbounds for j = 3:n, i = 1:n
        v = A[i,j] + y * A_up[i,j]
        if v > Y2[i]
            if v > Y1[i]
                Y2[i] = Y1[i]
                I1[i] = j
                Y1[i] = v
            else
                Y2[i] = v
            end
        end
    end

    # compute R values
    @inbounds for j = 1:n, i = 1:n
        mv = (j == I1[i] ? Y2[i] : Y1[i])
        R[i,j] = - mv + y * A_up[i,j]
    end

    return R
end

# compute availabilities
function _afp_compute_a!(A::Matrix{T}, R::Matrix{T}) where T
    n = size(R, 1)
    z = zero(T)
    for j = 1:n
        @inbounds rjj = R[j,j]

        # compute s <- sum_{i \ne j} max(0, R(i,j))
        s = z
        for i = 1:n
            if i != j
                @inbounds r = R[i,j]
                if r > z
                    s += r
                end
            end
        end

        for i = 1:n
            if i == j
                @inbounds A[i,j] = s
            else
                @inbounds r = R[i,j]
                u = rjj + s
                if r > z
                    u -= r
                end
                A[i,j] = ifelse(u < z, u, z)
            end
        end
    end
    return A
end

# compute interactions
function _afp_compute_a_up!(A_up::Matrix{T}, A::Matrix{T}, S::Matrix{T}, γ::T) where T
    n = size(S, 1)

    A_up .= S .+ A .+ γ
    v = maximum(A_up, dims=2) .- γ
    @inbounds for k = 1:n, i = 1:n
        A_up[i, k] = max(A_up[i, k], v[i])
    end

    return A_up
end

function _afp_compute_a_down!(A_down::Matrix{T}, A_ref::Matrix{T}, A_up::Matrix{T}, γ::T, y::T) where T
    n = size(A_ref, 1)

    A_down .= A_ref .+ (y-1) .* A_up .+ γ
    v = maximum(A_down, dims=2) .- γ
    @inbounds for k = 1:n, i = 1:n
        A_down[i, k] = max(A_down[i, k], v[i])
    end

    return A_down
end

# dampen update
function _afp_dampen_update!(x::Array{T}, xt::Array{T}, damp::T) where T
    ct = one(T) - damp
    for i = 1:length(x)
        @inbounds x[i] = ct * xt[i] + damp * x[i]
    end
    return x
end

# count the number of exemplars
function _afp_count_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    c = 0
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            c += 1
        end
    end
    return c
end

# extract all exemplars
function _afp_extract_exemplars(A::Matrix, R::Matrix)
    n = size(A,1)
    r = Int[]
    for i = 1:n
        @inbounds if A[i,i] + R[i,i] > 0
            push!(r, i)
        end
    end
    return r
end

# get assignments
function _afp_get_assignments(S::DenseMatrix, exemplars::Vector{Int})
    n = size(S, 1)
    k = length(exemplars)
    Se = S[:, exemplars]
    a = Vector{Int}(undef, n)
    cnts = zeros(Int, k)
    for i = 1:n
        p = 1
        v = Se[i,1]
        for j = 2:k
            s = Se[i,j]
            if s > v
                v = s
                p = j
            end
        end
        a[i] = p
    end
    for i = 1:k
        a[exemplars[i]] = i
    end
    for i = 1:n
        @inbounds cnts[a[i]] += 1
    end
    return (a, cnts)
end

function _afp_compute_energy(S::Matrix{Float64}, exemplars::Vector{Int}, assignments::Vector{Int})
    E = 0
    N = size(S,2)

    for i in 1:N
        E -= S[i,exemplars[assignments[i]]]
    end

    return E
end

function main(S::Matrix{Float64};
              seed::Int=-1,
              init::String="gauss",   # mess init: [:gauss, :unif]
              α::Float64=0.1,        # mess are init with N(o,α) or U(-α, α)
              dataset::String="",
              maxiter::Int=1000,
              damp::Float64=0.5,
              tol::Float64=1e-9,       # Update converged if max(mess(t) - mess(t-1)) < tol
              λ::Union{Nothing,Float64}=nothing,        # Self affinity parameter
              y::Float64=0.0,        # number of replicas. fAP is used whenever y > 0 || γ > 0
              yfact::Float64=0.0,    # y(t) = y * (1+yfact)^t
              γ::Float64=0.0,        # elastic interaction. fAP is used whenever y > 0 || γ > 0
              γfact::Float64=0.0,    # γ(t) = γ * (1+γfact)^t
              print::Bool=true,      # print every iter
              print_res::Bool=false, # print result only
              write::Bool=false,
              write_res::Bool=false,
              outfile::String=""
              )

    seed > 0 && Random.seed!(seed)

    # Maybe i dont need this
    !isempty(dataset) && (S = readdlm(dataset))
    N = size(S, 1)
    size(S, 2) == N || throw(ArgumentError("S must be a square matrix ($(size(S)) given)."))
    N >= 2 || throw(ArgumentError("At least two data points are required ($N given)."))

    if isa(λ, Float64)
        for i = 1:N
            S[i,i] = λ
        end
    end

    if γ > 0.0 || y > 0.0
        # focusing AP
        res = _affinitypropR(S; maxiter=maxiter, tol=tol, init=init, α=α, damp=damp,
                                print=print, print_res=print_res,
                                γ=γ, γfact=γfact, y=y, yfact=yfact)
    else
        # vanilla AP
        res = _affinityprop(S; maxiter=maxiter, tol=tol, init=init, α=α, damp=damp,
                               print=print, print_res=print_res)
    end

    return res
end # main

end # module
