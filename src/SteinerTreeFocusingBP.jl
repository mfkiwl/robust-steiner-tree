module SteinerTreeFocusingBP

# TODO:
# *) Choose Root (Is this useful only for PCST?)
# *) Prize collecting steiner tree
# *) Fix various issues: mess convergence, deg 1 nodes
# *) Use plotting function as a module (e.g. read from file and plot)
# *) Can I fix warning from Cairo and Compose?

using Random, Statistics, LinearAlgebra
using ExtractMacro, DelimitedFiles, Printf
using LightGraphs, GraphPlot
#using Gadfly, Colors, Cairo, Fontconfig
using Colors
#using Clustering

const F = Float64
const VF = Vector{F}
const VVF = Vector{VF}
const PF = Ptr{F}
const VPF = Vector{PF}

const myInf = F(1e10)

mutable struct VarNode
    Δ::Int
    d::Int
    p::Int
    neighs::Vector{Int}
    w::VF
    is_terminal::Bool
    # Messages
    Ain::Vector{VF}
    Aout::Vector{PF}
    Bin::VF
    Bout::Vector{PF}
    Din::VF
    Dout::Vector{PF}
    Ein::Vector{VF}
    Eout::Vector{PF}
    # Marginals
    Ψ::Vector{VF}
    Γ::F
    # Interaction messages
    ϕ::VF # distance coupling
    φ::VF # pointer coupling
    φ0::F # pointer coupling
end
VarNode(Δ) = VarNode(Δ, 0, 0, Int[], VF[], false,
                     VVF(), VPF(),      # A
                     VF(), VPF(),       # B
                     VF(), VPF(),       # D
                     VVF(), VPF(),      # E
                     VVF(), F(0),       # Ψ, Γ
                     VF(), VF(), F(0))  # ϕ, φ, φ0

deg(v::VarNode) = length(v.neighs)

mutable struct FactorGraph
    g::SimpleGraph{Int64}
    N::Int
    Δ::Int
    root_id::Int
    vnodes::Vector{VarNode}

    function FactorGraph(N::Int, Δ::Int, α::Float64;
                         graph_seed::Int = 123,
                         graph_type::Union{Symbol, String}=:fc,
                         term_file::String = "",
                         c::Int=10, # (average) connectivity
                         root_id::Int = 1,
                         distr::Symbol = :unif,
                         σ::Float64=1.0,
                         init::F=F(0))

        if isa(graph_type, String)
            input_graph = readdlm(graph_type)
            numEdges, col = size(input_graph)
            @assert col == 3 # i j wij  with i < j
            N = Int(maximum(input_graph[:,1:2]))

            g = Graph()
            add_vertices!(g, N)
            for e = 1:numEdges
                add_edge!(g, input_graph[e,1], input_graph[e,2])
            end
        elseif graph_type == :fc
            g = complete_graph(N)
        elseif graph_type == :er
            z = F(c) / F(N-1)
            g = erdos_renyi(N, z; seed=graph_seed)
        elseif graph_type == :rrg
            g = random_regular_graph(N, c, seed=graph_seed)
        else
            error("Wrong graph type. Got $graph_type. Options are [:fc, :er, :rrg]")
        end
        @assert root_id <= N

        vnodes = [VarNode(Δ) for i = 1:N]

        if !isempty(term_file)
            terms = readdlm(term_file)
        end

        for (i, v) in enumerate(vnodes)
            for j in neighbors(g, i)
                push!(v.neighs, j)
            end
            numNeighs = deg(v)

            if i == root_id
                @assert numNeighs > 0
                v.is_terminal = true # TODO: change this!
            end
            if isempty(term_file)
                rand() < α    && (v.is_terminal = true)
                numNeighs < 1 && (v.is_terminal = false)
            else
                terms[i] != 0 && (v.is_terminal = true)
            end

            v.w = zeros(F, numNeighs)
            resize!(v.Ain, numNeighs)
            resize!(v.Aout, numNeighs)
            resize!(v.Bin, numNeighs)
            resize!(v.Bout, numNeighs)
            resize!(v.Din, numNeighs)
            resize!(v.Dout, numNeighs)
            resize!(v.Ein, numNeighs)
            resize!(v.Eout, numNeighs)
            #
            resize!(v.Ψ, numNeighs)
        end

        for (i, v) in enumerate(vnodes)
            v.ϕ = zeros(F, Δ)
            v.φ = zeros(F, N)
            for (j, nn) in enumerate(v.neighs)
                v.Ain[j] = -init .* rand(F, Δ)
                v.Bin[j] = -init  * rand(F)
                v.Din[j] = -init  * rand(F)
                v.Ein[j] = -init .* rand(F, Δ)
                #
                v.Ψ[j] = zeros(F, Δ)

                k = findfirst(isequal(i), vnodes[nn].neighs)

                vnodes[nn].Aout[k] = pointer(v.Ain[j], 1)
                vnodes[nn].Bout[k] = pointer(v.Bin, j)
                vnodes[nn].Dout[k] = pointer(v.Din, j)
                vnodes[nn].Eout[k] = pointer(v.Ein[j], 1)
            end
        end

        if isa(graph_type, String)
            for e = 1:numEdges
                i::Int, nn::Int, r = input_graph[e, :]
                j = findfirst(isequal(nn), vnodes[i].neighs)
                vnodes[i].w[j] = r
                k = findfirst(isequal(i), vnodes[nn].neighs)
                vnodes[nn].w[k] = r
            end
        else
            init_rand_w(vnodes; distr=distr, σ=σ, seed=graph_seed)
        end

        new(g, N, Δ, root_id, vnodes)
    end

end # struct FactorGraph

function init_rand_w(vnodes::Vector{VarNode}; distr::Symbol=:unif, σ::F=1.0, seed::Int=0)
    seed > 0 && Random.seed!(seed)

    for (i, v) in enumerate(vnodes)
        for (j, nn) in enumerate(v.neighs)
            (i > nn) && continue
            if distr == :unif
                r = σ * rand()
            elseif distr == :exp
                r = σ * randexp() #?
            else
                error("Please choose distr among [:unif, :exp]")
            end
            v.w[j] = r
            k = findfirst(isequal(i), vnodes[nn].neighs)
            vnodes[nn].w[k] = r
        end
    end
end

# TODO: find a better way!
include("STFBPutils.jl")

function update_root!(v::VarNode)
    @extract v Aout Bout Dout Eout Δ w

    for j = 1:deg(v)
        unsafe_store!(Bout[j], -myInf)
        unsafe_store!(Dout[j], F(0))
        for d = 1:Δ
            unsafe_store!(Eout[j], F(0), d)
            unsafe_store!(Aout[j], -myInf, d)
        end
    end
end

# For nodes with connectivity one. Not nice :(
function update_leave!(v::VarNode; root_id::Int=1, ρ::Float64=0.0)
    @extract v Aout Ain Bout Dout Din Eout Ein Ψ Γ is_terminal w Δ neighs

    B = is_terminal ? -myInf : ρ * Γ
    unsafe_store!(Bout[1], B)

    maxA = -myInf
    for d = 1:Δ
        A = -myInf
        unsafe_store!(Aout[1], A, d)
        maxA = max(maxA, A)
    end

    D = max(B, maxA)
    unsafe_store!(Dout[1], D)
    C = -myInf
    for d = Δ:-1:1
        E = max(C, D)
        unsafe_store!(Eout[1], E, d)
        C = -w[1] + ρ * Ψ[1][d]
        if d == 1
            Ψ[1][d] = v.neighs[1] == root_id ? C : -myInf
        else
            Ψ[1][d] = C + Ain[1][d-1]
        end
    end
    v.Γ = is_terminal ? -myInf : Din[1] + ρ * v.Γ

    normalize_mess!(v, 1)

end

function update_mess!(v::VarNode; root_id::Int=1, ρ::Float64=0.0)
    @extract v w Ain Aout Bout Din Dout Ein Eout Ψ Γ Δ is_terminal

    sumE = zeros(F, Δ)
    sumD = F(0)
    maxCav = -Inf .* ones(F, Δ)
    maxCav2 = -Inf .* ones(F, Δ)
    # maxCav = -3.0*myInf .* ones(F, Δ)
    # maxCav2 = -3.0*myInf .* ones(F, Δ)
    maxCavIdx = -1.0 .* ones(F, Δ)

    for j = 1:deg(v)
        for d = 1:Δ
            sumE[d] += Ein[j][d]
            if d == 1
                # Ain[Root][0] = 0
                M = (v.neighs[j] == root_id) ? (- w[j] - Ein[j][d] + ρ * Ψ[j][d]) : -myInf
            else
                M = - w[j] - Ein[j][d] + Ain[j][d-1] + ρ * Ψ[j][d]
            end

            if maxCav[d] <= M
                maxCav2[d] = maxCav[d]
                maxCav[d] = M
                maxCavIdx[d] = j
            elseif maxCav2[d] < M
                maxCav2[d] = M
            end
        end
        sumD += Din[j]
    end

    for j = 1:deg(v)
        maxA = -myInf
        newB = is_terminal ? -myInf : sumD - Din[j] + ρ * Γ
        unsafe_store!(Bout[j], newB)

        for d = 1:Δ
            @assert maxCavIdx[d] > 0
            M = maxCavIdx[d] == j ? maxCav2[d] : maxCav[d]
            newA = sumE[d] - Ein[j][d] + M
            unsafe_store!(Aout[j], newA, d)
            maxA = max(maxA, newA)
        end
        newD = max(maxA, newB)
        unsafe_store!(Dout[j], newD)

        C = -myInf
        for d = Δ:-1:1
            newE = max(C, newD)
            unsafe_store!(Eout[j], newE, d)
            C = - w[j] + sumE[d] - Ein[j][d] + ρ * Ψ[j][d]
            Ψ[j][d] = (d == 1) ? (v.neighs[j] == root_id ? C : -myInf) : (C + Ain[j][d-1])
        end

        normalize_mess!(v, j)
    end # for j

    v.Γ = is_terminal ? -myInf : sumD + ρ * Γ

end

# FBP update
########################
# distance interaction #
########################
function update_mess_dInt!(v::VarNode, ζ::VF; root_id::Int=1, γ::Float64=0.0)
    @extract v w Ain Aout Bout Din Dout Ein Eout Ψ Γ ϕ Δ is_terminal

    sumE = zeros(F, Δ)
    H = zeros(F, Δ)
    sumD = F(0)
    maxCav = -Inf .* ones(F, Δ)
    maxCav2 = -Inf .* ones(F, Δ)
    maxCavIdx = -1.0 .* ones(F, Δ)

    for j = 1:deg(v)
        for d = 1:Δ
            sumE[d] += Ein[j][d]
            if d == 1
                M = (v.neighs[j] == root_id) ? (- w[j] - Ein[j][d]) : -myInf
            else
                M = - w[j] - Ein[j][d] + Ain[j][d-1]
            end
            if maxCav[d] <= M
                maxCav2[d] = maxCav[d]
                maxCav[d] = M
                maxCavIdx[d] = j
            elseif maxCav2[d] < M
                maxCav2[d] = M
            end
        end
        sumD += Din[j]
    end

    maxζ = -myInf
    for d = 1:Δ
        maxζ = max(maxζ, ζ[d])
    end

    for j = 1:deg(v)
        maxA = -myInf
        newB = is_terminal ? -myInf : sumD - Din[j] + maxζ
        unsafe_store!(Bout[j], newB)

        for d = 1:Δ
            @assert maxCavIdx[d] > 0
            M = maxCavIdx[d] == j ? maxCav2[d] : maxCav[d]
            newA = sumE[d] - Ein[j][d] + M + ζ[d]
            unsafe_store!(Aout[j], newA, d)
            maxA = max(maxA, newA)
        end
        newD = max(maxA, newB)
        unsafe_store!(Dout[j], newD)

        C = -myInf
        for d = Δ:-1:1
            newE = max(C, newD)
            unsafe_store!(Eout[j], newE, d)
            C = - w[j] + sumE[d] - Ein[j][d] + ζ[d]
            Ψ[j][d] = (d == 1) ? (v.neighs[j] == root_id ? C : -myInf) : (C + Ain[j][d-1])
        end

        normalize_mess!(v, j)
    end # for j

    maxH = -myInf
    maxH2 = -myInf
    maxHidx = -1
    for d = 1:Δ
        maxAE = -myInf
        for j = 1:deg(v)
            AE = (d == 1) ?
                 (v.neighs[j] == root_id ? - w[j] - Ein[j][d] : -myInf) :
                 (- w[j] - Ein[j][d] + Ain[j][d-1])
            maxAE = max(maxAE, AE)
        end

        H[d] = maxAE + sumE[d] - γ

        if maxH <= H[d]
            maxH2 = maxH
            maxH = H[d]
            maxHidx = d
        elseif maxH2 < H[d]
            maxH2 = H[d]
        end
    end
    ϕ0 = is_terminal ? -myInf : sumD
    for d = 1:Δ
        if d == maxHidx
            ϕ[d] = max(H[d] + γ, maxH2)
        else
            ϕ[d] = max(H[d] + γ, maxH)
        end
        ϕ[d] = max(ϕ[d], ϕ0)
    end

    v.Γ = is_terminal ? -myInf : sumD + maxζ
end

function update_ref_mess_dInt!(v::VarNode, ζ::VF; root_id::Int=1, γ::Float64=0.0, y::Float64=0.0)
    @extract v w Ain Aout Bout Din Dout Ein Eout Ψ Γ ϕ Δ is_terminal

    sumE = zeros(F, Δ)
    H = zeros(F, Δ)
    sumD = F(0)
    maxCav = -Inf .* ones(F, Δ)
    maxCav2 = -Inf .* ones(F, Δ)
    maxCavIdx = -1.0 .* ones(F, Δ)

    for j = 1:deg(v)
        for d = 1:Δ
            sumE[d] += Ein[j][d]
            if d == 1
                M = (v.neighs[j] == root_id) ? (- Ein[j][d]) : -myInf
            else
                M = - Ein[j][d] + Ain[j][d-1]
            end
            if maxCav[d] <= M
                maxCav2[d] = maxCav[d]
                maxCav[d] = M
                maxCavIdx[d] = j
            elseif maxCav2[d] < M
                maxCav2[d] = M
            end
        end
        sumD += Din[j]
    end

    maxζ = -myInf
    for d = 1:Δ
        maxζ = max(maxζ, ζ[d])
    end

    for j = 1:deg(v)
        maxA = -myInf
        newB = is_terminal ? -myInf : sumD - Din[j] + y * maxζ
        unsafe_store!(Bout[j], newB)

        for d = 1:Δ
            @assert maxCavIdx[d] > 0
            M = maxCavIdx[d] == j ? maxCav2[d] : maxCav[d]
            newA = sumE[d] - Ein[j][d] + M + y * ζ[d]
            unsafe_store!(Aout[j], newA, d)
            maxA = max(maxA, newA)
        end
        newD = max(maxA, newB)
        unsafe_store!(Dout[j], newD)

        C = -myInf
        for d = Δ:-1:1
            newE = max(C, newD)
            unsafe_store!(Eout[j], newE, d)
            C = sumE[d] - Ein[j][d] + y * ζ[d]
        end

        normalize_mess!(v, j)
    end # for j

    maxH = -myInf
    maxH2 = -myInf
    maxHidx = -1
    for d = 1:Δ
        maxAE = -myInf
        for j = 1:deg(v)
            AE = (d == 1) ?
                 (v.neighs[j] == root_id ? - Ein[j][d] : -myInf) :
                 (- Ein[j][d] + Ain[j][d-1])
            maxAE = max(maxAE, AE)
        end

        H[d] = maxAE + sumE[d] + (y-1) * ζ[d] - γ
        H0 = is_terminal ? -myInf : (sumD + (y-1) * ζ[d] - γ)

        H[d] = max(H[d], H0)

        if maxH <= H[d]
            maxH2 = maxH
            maxH = H[d]
            maxHidx = d
        elseif maxH2 < H[d]
            maxH2 = H[d]
        end
    end
    for d = 1:Δ
        if d == maxHidx
            ϕ[d] = max(H[d] + γ, maxH2)
        else
            ϕ[d] = max(H[d] + γ, maxH)
        end
    end

end

#######################
# pointer interaction #
#######################
function update_mess_pInt!(v::VarNode, ζ::VF, ζ0::F; root_id::Int=1, γ::Float64=0.0)
    @extract v w Ain Aout Bout Din Dout Ein Eout Ψ Γ φ Δ is_terminal

    sumE = zeros(F, Δ)
    H = zeros(F, deg(v))
    sumD = F(0)
    maxCav = -Inf .* ones(F, Δ)
    maxCav2 = -Inf .* ones(F, Δ)
    maxCavIdx = -1.0 .* ones(F, Δ)

    for j = 1:deg(v)
        for d = 1:Δ
            sumE[d] += Ein[j][d]
            if d == 1
                M = (v.neighs[j] == root_id) ? (- w[j] - Ein[j][d] + ζ[j]) : -myInf
            else
                M = - w[j] - Ein[j][d] + Ain[j][d-1] + ζ[j]
            end
            if maxCav[d] <= M
                maxCav2[d] = maxCav[d]
                maxCav[d] = M
                maxCavIdx[d] = j
            elseif maxCav2[d] < M
                maxCav2[d] = M
            end
        end
        sumD += Din[j]
    end

    for j = 1:deg(v)
        maxA = -myInf
        newB = is_terminal ? -myInf : sumD - Din[j] + ζ0
        unsafe_store!(Bout[j], newB)

        for d = 1:Δ
            @assert maxCavIdx[d] > 0
            M = maxCavIdx[d] == j ? maxCav2[d] : maxCav[d]
            newA = sumE[d] - Ein[j][d] + M
            unsafe_store!(Aout[j], newA, d)
            maxA = max(maxA, newA)
        end
        newD = max(maxA, newB)
        unsafe_store!(Dout[j], newD)

        C = -myInf
        for d = Δ:-1:1
            newE = max(C, newD)
            unsafe_store!(Eout[j], newE, d)
            C = - w[j] + sumE[d] - Ein[j][d] + ζ[j]
            Ψ[j][d] = (d == 1) ? (v.neighs[j] == root_id ? C : -myInf) : (C + Ain[j][d-1])
        end

        normalize_mess!(v, j)
    end # for j

    maxH = -myInf
    maxH2 = -myInf
    maxHidx = -1
    for j = 1:deg(v)
        maxAE = -myInf

        AE = v.neighs[j] == root_id ? sumE[1] - Ein[j][1] : -myInf
        maxAE = max(AE, maxAE)
        for d = 2:Δ
            AE = sumE[d] - Ein[j][d] + Ain[j][d-1]
            maxAE = max(AE, maxAE)
        end

        H[j] = - w[j] + maxAE - γ

        if maxH <= H[j]
            maxH2 = maxH
            maxH = H[j]
            maxHidx = j
        elseif maxH2 < H[j]
            maxH2 = H[j]
        end
    end

    for j = 1:deg(v)
        if j == maxHidx
            φ[j] = max(H[j] + γ, maxH2)
        else
            φ[j] = max(H[j] + γ, maxH)
        end
        φ[j] = max(φ[j], (is_terminal ? -myInf : - γ + sumD))
    end
    v.φ0 = max((is_terminal ? -myInf : sumD), maxH)

    v.Γ = is_terminal ? -myInf : sumD + ζ0
end

function update_ref_mess_pInt!(v::VarNode, ζ::VF, ζ0::F; root_id::Int=1, γ::Float64=0.0, y::Float64=0.0)
    @extract v w Ain Aout Bout Din Dout Ein Eout Ψ Γ φ Δ is_terminal

    sumE = zeros(F, Δ)
    H = zeros(F, deg(v))
    sumD = F(0)
    maxCav = -Inf .* ones(F, Δ)
    maxCav2 = -Inf .* ones(F, Δ)
    maxCavIdx = -1.0 .* ones(F, Δ)

    for j = 1:deg(v)
        for d = 1:Δ
            sumE[d] += Ein[j][d]
            if d == 1
                M = (v.neighs[j] == root_id) ? (- Ein[j][d] + y * ζ[j]) : -myInf
            else
                M = - Ein[j][d] + Ain[j][d-1] + y * ζ[j]
            end
            if maxCav[d] <= M
                maxCav2[d] = maxCav[d]
                maxCav[d] = M
                maxCavIdx[d] = j
            elseif maxCav2[d] < M
                maxCav2[d] = M
            end
        end
        sumD += Din[j]
    end

    for j = 1:deg(v)
        maxA = -myInf
        newB = is_terminal ? -myInf : sumD - Din[j] + y * ζ0
        unsafe_store!(Bout[j], newB)

        for d = 1:Δ
            @assert maxCavIdx[d] > 0
            M = maxCavIdx[d] == j ? maxCav2[d] : maxCav[d]
            newA = sumE[d] - Ein[j][d] + M
            unsafe_store!(Aout[j], newA, d)
            maxA = max(maxA, newA)
        end
        newD = max(maxA, newB)
        unsafe_store!(Dout[j], newD)

        C = -myInf
        for d = Δ:-1:1
            newE = max(C, newD)
            unsafe_store!(Eout[j], newE, d)
            C = sumE[d] - Ein[j][d] + y * ζ[j]
        end

        normalize_mess!(v, j)
    end # for j

    maxH = -myInf
    maxH2 = -myInf
    maxHidx = -1
    for j = 1:deg(v)
        maxAE = -myInf
        AE = v.neighs[j] == root_id ? sumE[1] - Ein[j][1] : -myInf
        maxAE = max(AE, maxAE)
        for d = 2:Δ
            AE = sumE[d] - Ein[j][d] + Ain[j][d-1]
            maxAE = max(AE, maxAE)
        end

        H[j] = maxAE + (y-1) * ζ[j] - γ

        if maxH <= H[j]
            maxH2 = maxH
            maxH = H[j]
            maxHidx = j
        elseif maxH2 < H[j]
            maxH2 = H[j]
        end
    end

    for j = 1:deg(v)
        if j == maxHidx
            φ[j] = max(H[j] + γ, maxH2)
        else
            φ[j] = max(H[j] + γ, maxH)
        end
        φ[j] = max(φ[j], (is_terminal ? -myInf : - γ + sumD + (y-1) * ζ0))
    end
    v.φ0 = max((is_terminal ? -myInf : sumD + (y-1) * ζ0), maxH)

end


function normalize_mess!(v::VarNode, j::Int)
    @extract v Aout Bout Dout Eout w Δ

    M = -Inf
    B = unsafe_load(Bout[j])
    D = unsafe_load(Dout[j])
    M = max(M, B, D)
    for d = 1:Δ
        A = unsafe_load(Aout[j], d)
        E = unsafe_load(Eout[j], d)
        M = max(M, A, E)
    end

    #@assert isfinite(M)
    if isfinite(M)
        B = unsafe_load(Bout[j])
        D = unsafe_load(Dout[j])
        isfinite(B) && unsafe_store!(Bout[j], B - M)
        unsafe_store!(Dout[j], D - M)
        for d = 1:Δ
            A = unsafe_load(Aout[j], d)
            E = unsafe_load(Eout[j], d)
            isfinite(A) && unsafe_store!(Aout[j], A - M, d)
            unsafe_store!(Eout[j], E - M, d)
        end
    end

end

function oneBPstep!(G::FactorGraph; ρ::Float64=0.0)
    @extract G N vnodes root_id

    # TODO: root needs to be updated once
    update_root!(vnodes[root_id])
    for i in randperm(N)
        i == root_id && continue
        deg(vnodes[i]) == 0 && continue
        deg(vnodes[i]) == 1 && update_leave!(vnodes[i]; root_id=root_id, ρ=ρ)
        deg(vnodes[i]) >  1 && update_mess!(vnodes[i]; root_id=root_id, ρ=ρ)
        #update_mess!(vnodes[i]; root_id=root_id, ρ=ρ)
        # TODO: fix the deg 1 case in update_mess
    end
end

function oneFBPstep_dInt!(G::FactorGraph, Gref::FactorGraph;
                          γ::Float64=0.0, y::Float64=0.0)
    @extract G N root_id
    # TODO: root needs to be updated once
    update_root!(G.vnodes[root_id])
    update_root!(Gref.vnodes[root_id])
    for i in randperm(N)
        i == root_id && continue
        deg(G.vnodes[i]) == 0 && continue
        update_mess_dInt!(G.vnodes[i], Gref.vnodes[i].ϕ; root_id=root_id, γ=γ)
        update_ref_mess_dInt!(Gref.vnodes[i], G.vnodes[i].ϕ; root_id=root_id, γ=γ, y=y)
    end
end

function oneFBPstep_pInt!(G::FactorGraph, Gref::FactorGraph;
                          γ::Float64=0.0, y::Float64=0.0)
    @extract G N root_id
    # TODO: root needs to be updated once
    update_root!(G.vnodes[root_id])
    update_root!(Gref.vnodes[root_id])
    for i in randperm(N)
        i == root_id && continue
        deg(G.vnodes[i]) == 0 && continue
        update_mess_pInt!(G.vnodes[i], Gref.vnodes[i].φ, Gref.vnodes[i].φ0; root_id=root_id, γ=γ)
        update_ref_mess_pInt!(Gref.vnodes[i], G.vnodes[i].φ, G.vnodes[i].φ0; root_id=root_id, γ=γ, y=y)
    end
end

function assign_variables!(v::VarNode; root_id::Int=1)
    @extract v Γ Ψ Δ

    M, v.d, v.p = Γ, -1, -1
    #M = -Inf
    for d = 1:Δ
        for j = 1:deg(v)
            if Ψ[j][d] >= M
                M = Ψ[j][d]
                v.p = v.neighs[j]
                v.d = d
            end
        end
    end
    # if Γ > M
    #     v.p, v.d = -1, -1
    # end
end

function assign_variables!(G::FactorGraph)
    @extract G vnodes root_id
    for (i, v) in enumerate(vnodes)
        if i == root_id
            v.d, v.p = 0, 1
        else
            assign_variables!(v; root_id=root_id)
        end
    end
end

function is_good(G::FactorGraph; verbose::Bool=false)
    @extract G N Δ vnodes root_id

    for i = 1:N
        i == root_id && continue
        pi, di = vnodes[i].p, vnodes[i].d

        @assert di < Δ + 1
        @assert pi < N + 1
        if pi < 0
            # terminals must be in the tree
            if vnodes[i].is_terminal && deg(vnodes[i]) > 0
                verbose && println("ERROR: node $i is terminal but isnt in the tree")
                return false
            end
        else
            # the variable you point should point someone
            if vnodes[pi].p < 0
                return false
                verbose && println("ERROR: node $i points to a non valid site")
            end
            # your depth must be one less than the depth of the site you are pointing
            if !(vnodes[pi].d == di - 1)
                return false
                verbose && println("ERROR: node $i with d = $di points to $pi with d = $(vnodes[pi].d)")
            end
        end
    end
    return true
end

function cost(G::FactorGraph)
    @extract G N vnodes root_id

    E = 0.0
    for i = 1:N
        i == root_id && continue
        pi = vnodes[i].p
        if pi > 0
            j = findfirst(isequal(pi), vnodes[i].neighs)
            E += vnodes[i].w[j]
        end
    end
    return E
end

function solution_weights(G::FactorGraph)
    @extract G N vnodes root_id

    w = []
    for i = 1:N
        i == root_id && continue
        pi = vnodes[i].p
        if pi > 0
            j = findfirst(isequal(pi), vnodes[i].neighs)
            push!(w, vnodes[i].w[j])
        end
    end
    return w
end

function main(N::Int, Δ::Int, α::Float64;
              seed::Int = 0,
              graph_seed::Int = 0,       # fixes random graph and weights generation
              graph::Union{Symbol, String} = :fc, # [:fc, :er, :rrg]
              term_file::String = "",    # N rows file, if entry != 0 node is terminal
              root_id::Int = 1,          # which site is the root
              c::Int = 3,                # er/rrg average connectivity
              σ::Float64 = 1.0,          # weights init as σ * rand()
              distr::Symbol = :unif,     # weights distr [:unif, :exp]
              mess_init::F = F(0),       # mess init as σ * rand()
              maxiter::Int = 100,
              tconv::Int = 10,           # stop if valid sol if found tconv consecutive steps
              #
              ρ::Float64 = 0.0,          # reinforcement ρ(t) = ρ + t*ρsteps
              ρstep::Float64 = 0.0,      # reinforcement ρ(t) = ρ + t*ρsteps
              γ::Float64 = 0.0,
              γstep::Float64 = 0.0,
              γfact::Float64 = 0.0,
              y::Float64 = 0.0,
              ystep::Float64 = 0.0,
              interaction::Symbol = :dist, # [:dist, :pointer]
              #
              compute_mst::Bool = false, # compute min spannig tree
              kmst_out::String = "",     # plot Kruskal min spann tree
              pmst_out::String = "",     # plot Prim min spann tree
              graph_out::String = "",    # plot the graph instance
              sol_out::String = "",      # plot the solution tree
              verbose::Bool = true)      # print iters on screen

    seed > 0 && Random.seed!(seed)

    # Generate the instance (or read it from a file)
    # Use graph_seed to fix the random instance (graph+weights)
    G = FactorGraph(N, Δ, α; graph_type=graph, term_file=term_file, graph_seed=graph_seed, c=c, σ=σ, distr=distr, root_id=root_id, init=mess_init)
    if (γ > 0.0 || y > 0.0)
        #@assert y > 1
        Gref = FactorGraph(N, Δ, α; graph_type=graph, term_file=term_file, graph_seed=graph_seed, c=c, σ=σ, distr=distr, root_id=root_id, init=mess_init)
    end

    # Plot the instance
    if !isempty(graph_out)
        plot_graph(G, graph_out)
    end

    # Compute Minimum Spanning Tree, end eventually plot it
    # Useful as a reference, provides the best cost for α = 1, Δ ≧ N-1
    if compute_mst
        kruskal_edges, prim_edges = min_span_tree_cost(G)
        !isempty(kmst_out) && plot_sol_tree(G, kmst_out, kruskal_edges)
        !isempty(pmst_out) && plot_sol_tree(G, pmst_out, prim_edges)
    end

    iter = 0
    correct_steps = 0
    for t = 1:maxiter
        if γ > 0.0 || y > 0.0
            if interaction == :dist
                oneFBPstep_dInt!(G, Gref; γ=γ, y=y)
            elseif interaction == :pointer
                oneFBPstep_pInt!(G, Gref; γ=γ, y=y)
            end
        else
            oneBPstep!(G; ρ=ρ)
        end
        assign_variables!(G)
        is_good(G) ? (correct_steps += 1) : (correct_steps = 0)

        s = @sprintf("iter=%i, correct_steps=%i", t, correct_steps)
        ρ > 0.0 && (s *= @sprintf(", ρ=%3.3f ", ρ))
        (γ > 0.0 || y > 0.0) && (s *= @sprintf(", γ=%3.3f, y=%3.3f ", γ, y))
        verbose && print("\r$s")

        correct_steps == tconv && break
        ρ += ρstep
        γ += γstep
        γ *= (1.0 + γfact)
        y += ystep
        iter = t # sad :(
    end

    E = cost(G)
    converged = false
    if correct_steps == tconv
        verbose && @info "Final cost = $E"
        converged = true
    end

    # Plot the solution tree
    if !isempty(sol_out)
        plot_sol_tree(G, sol_out)
    end

    terminal_nodes = [G.vnodes[i].is_terminal for i = 1:(G.N)]
    p = [G.vnodes[i].p for i = 1:(G.N)]
    d = [G.vnodes[i].d for i = 1:(G.N)]
    w = solution_weights(G)

    num_steiner = 0
    for i = 1:(G.N)
        if G.vnodes[i].is_terminal == false && G.vnodes[i].p > 0
            num_steiner += 1
        end
    end

    return converged, iter, E, num_steiner, w, terminal_nodes, p, d
end

end # module
