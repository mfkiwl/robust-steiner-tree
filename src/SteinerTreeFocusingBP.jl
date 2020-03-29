module SteinerTreeFocusingBP

# TODO:
# *) Choose Root (Is this useful only for PCST?)
# *) Prize collecting steiner tree
# *) Read wi0 from file
# *) Check AP is ok
# *) Fix various issues: mess convergence, deg 1 nodes, reinf, ecc.
# *) Implement FBP
# *) Move plotting functions to another file
# *) Can I fix warning from Cairo and Compose?
# *) Cleaner seed options

using Random, Statistics, LinearAlgebra
using ExtractMacro, DelimitedFiles, Printf
using LightGraphs, GraphPlot
using Gadfly, Colors, Cairo, Fontconfig
#using Clustering

const F = Float64
const VF = Vector{F}
const VVF = Vector{VF}
const PF = Ptr{F}
const VPF = Vector{PF}

const myInf = F(1e10)
#const myInf = Inf

mutable struct VarNode
    Δ::Int
    d::Int
    p::Int
    neighs::Vector{Int}
    w::VF
    is_terminal::Bool
    #
    Ain::Vector{VF}
    Aout::Vector{PF}
    Bin::VF
    Bout::Vector{PF}
    Din::VF
    Dout::Vector{PF}
    Ein::Vector{VF}
    Eout::Vector{PF}
    #
    Ψ::Vector{VF}
    Γ::F
end
VarNode(Δ) = VarNode(Δ, 0, 0, Int[], VF[], false,
                     VVF(), VPF(),      # A
                     VF(), VPF(),       # B
                     VF(), VPF(),       # D
                     VVF(), VPF(),      # E
                     VVF(), F(-myInf))  # Ψ, Γ

deg(v::VarNode) = length(v.neighs)

mutable struct FactorGraph
    g::SimpleGraph{Int64}
    N::Int
    Δ::Int
    root_id::Int
    vnodes::Vector{VarNode}

    function FactorGraph(N::Int, Δ::Int, α::Float64;
                         graph_seed::Int = 123,
                         graph_type::Symbol=:fc,
                         c::Int=10, # (average) connectivity
                         root_id::Int = 1,
                         σ::Float64=1.0,
                         init::F=F(0))

        @assert root_id <= N

        if graph_type == :fc
            g = complete_graph(N)
        elseif graph_type == :er
            z = F(c) / F(N-1)
            g = erdos_renyi(N, z; seed=graph_seed)
        elseif graph_type == :rrg
            g = random_regular_graph(N, c, seed=graph_seed)
        else
            error("Wrong graph type. Got $graph_type. Options are [:fc, :er, :rrg]")
        end

        vnodes = [VarNode(Δ) for i = 1:N]

        for (i, v) in enumerate(vnodes)
            for j in neighbors(g, i)
                push!(v.neighs, j)
            end
            numNeighs = deg(v)

            if i == root_id
                @assert numNeighs > 0
                v.is_terminal = true # TODO: change this!
            end
            rand() < α && (v.is_terminal = true)
            numNeighs < 1 && (v.is_terminal = false)

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
        init_rand_w(vnodes, σ; seed=graph_seed)

        new(g, N, Δ, root_id, vnodes)
    end

    function FactorGraph(Δ::Int, α::Float64, graph_file::String;
                         root_id::Int=1,
                         init::F=F(0))

        input_graph = readdlm(graph_file)
        numEdges, col = size(input_graph)
        @assert col == 3 # i j wij  with i < j
        N = Int(maximum(input_graph[:,1:2]))

        g = Graph()
        add_vertices!(g, N)
        for e = 1:numEdges
            add_edge!(g, input_graph[e,1], input_graph[e,2])
        end

        vnodes = [VarNode(Δ) for i = 1:N]

        for (i, v) in enumerate(vnodes)
            for j in neighbors(g, i)
                push!(v.neighs, j)
            end
            numNeighs = deg(v)

            if i == root_id
                @assert numNeighs > 0
                v.is_terminal = true # TODO: change this!
            end
            rand() < α && (v.is_terminal = true)
            numNeighs < 1 && (v.is_terminal = false)

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

        for e = 1:numEdges
            i::Int, nn::Int, r = input_graph[e, :]
            j = findfirst(isequal(nn), vnodes[i].neighs)
            vnodes[i].w[j] = r
            k = findfirst(isequal(i), vnodes[nn].neighs)
            vnodes[nn].w[k] = r
        end

        for (i, v) in enumerate(vnodes)
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

        new(g, N, Δ, root_id, vnodes)
    end
end

function plot_graph(G::FactorGraph, graph_out::String)
    @extract G N g vnodes root_id

    terminals = [vnodes[i].is_terminal ? 1 : 2 for i = 1:N]
    nodecolor = [colorant"orange", colorant"lightgrey"]
    nodefillc = nodecolor[terminals]
    nodefillc[root_id] = colorant"red"

    draw(PNG(graph_out, 12cm, 12cm),
         gplot(g, nodelabel=1:N, nodefillc=nodefillc, edgelinewidth=edge_weights(vnodes)))

end

function plot_sol_tree(G::FactorGraph, graph_out::String)
    @extract G N g vnodes root_id

    terminals = [vnodes[i].is_terminal ? 1 : 2 for i = 1:N]
    nodecolor = [colorant"orange", colorant"lightgrey"]
    nodefillc = nodecolor[terminals]
    nodefillc[root_id] = colorant"red"

    edgecolors = [colorant"transparent" for e in 1:ne(g)]

    for (e, edge) in enumerate(edges(g))
        i = edge.src
        nn = edge.dst
        if vnodes[i].p == nn || vnodes[nn].p == i
            edgecolors[e] = colorant"green"
        end
    end

    draw(PNG(graph_out, 12cm, 12cm),
         gplot(g, nodelabel=1:N, nodefillc=nodefillc,
                  layout=circular_layout,
                  #edgelinewidth=edge_weights(vnodes),
                  edgestrokec=edgecolors))

end

function plot_sol_tree(G::FactorGraph, graph_out::String, edgelist)
    @extract G N g vnodes root_id

    terminals = [vnodes[i].is_terminal ? 1 : 2 for i = 1:N]
    nodecolor = [colorant"orange", colorant"lightgrey"]
    nodefillc = nodecolor[terminals]
    nodefillc[root_id] = colorant"red"

    edgecolors = [colorant"transparent" for e in 1:ne(g)]

    for (e, edge) in enumerate(edges(g))
        for l = 1:length(edgelist)
            if edgelist[l] == reverse(edge) || edgelist[l] == edge
                edgecolors[e] = colorant"green"
            end
        end
    end

    draw(PNG(graph_out, 12cm, 12cm),
         gplot(g, nodelabel=1:N, nodefillc=nodefillc,
                  layout=circular_layout,
                  #edgelinewidth=edge_weights(vnodes),
                  edgestrokec=edgecolors))

end

function edge_weights(vnodes::Vector{VarNode})
    N = length(vnodes)
    edgew = []
    for i = 1:N
        for j = 1:deg(vnodes[i])
            if i < vnodes[i].neighs[j]
                push!(edgew, vnodes[i].w[j])
            end
        end
    end
    return edgew
end

function min_span_tree_cost(G::FactorGraph)
    @extract G g N vnodes root_id

    distmx = ones(N, N)
    for (i, v) in enumerate(vnodes)
        for (j, nn) in enumerate(v.neighs)
            v.w[j] > 1.0 && @warn "You have weights > 1, mst cost will be wrong"
            distmx[i,nn] = v.w[j]
        end
    end

    # Kruskal MST
    kruskal_edges = kruskal_mst(g, distmx)
    kruskal_cost = 0.0
    for edge in kruskal_edges
        i, j = edge.dst, edge.src
        k = findfirst(isequal(j), vnodes[i].neighs)
        kruskal_cost += vnodes[i].w[k]
    end
    @info "Min Spanning Tree (Kruskal) cost = $kruskal_cost"

    # Prim MST
    prim_edges = prim_mst(g, distmx)
    prim_cost = 0.0
    for edge in prim_edges
        i, j = edge.dst, edge.src
        k = findfirst(isequal(j), vnodes[i].neighs)
        prim_cost += vnodes[i].w[k]
    end
    @info "Min Spanning Tree (Prim)    cost = $prim_cost"

    return kruskal_edges, prim_edges # needed to plot the sol
end

function init_rand_w(vnodes::Vector{VarNode}, σ::Float64; seed::Int=0)
    seed > 0 && Random.seed!(seed)

    for (i, v) in enumerate(vnodes)
        for (j, nn) in enumerate(v.neighs)
            (i > nn) && continue
            r = σ * rand()
            v.w[j] = r
            k = findfirst(isequal(i), vnodes[nn].neighs)
            vnodes[nn].w[k] = r
        end
    end
end

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
    Γ = is_terminal ? -myInf : Din[1] + ρ * Γ

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

    Γ = is_terminal ? -myInf : sumD + ρ * Γ

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

function assign_variables!(v::VarNode; root_id::Int=1)
    @extract v Γ Ψ Δ

    #M, di, pi = Γ, -1, -1
    M = -Inf
    for d = 1:Δ
        for j = 1:deg(v)
            if Ψ[j][d] >= M
                M = Ψ[j][d]
                v.p = v.neighs[j]
                v.d = d
            end
        end
    end
    if Γ > M
        v.p, v.d = -1, -1
    end
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

function main(N::Int, Δ::Int, α::Float64;
              seed::Int = 0,
              graph_seed::Int = 0,       # fixes random graph and weights generation
              graph::Union{Symbol, String} = :fc, # [:fc, :er, :rrg]
              root_id::Int = 1,          # which site is the root
              c::Int = 3,                # er/rrg average connectivity
              σ::Float64 = 1.0,          # weights init as σ * rand()
              mess_init::F = F(0),       # mess init as σ * rand()
              maxiter::Int = 100,
              tconv::Int = 10,           # stop if valid sol if found tconv consecutive steps
              ρ::Float64 = 0.0,          # reinforcement ρ(t) = ρ + t*ρsteps
              ρstep::Float64 = 0.0,      # reinforcement ρ(t) = ρ + t*ρsteps
              compute_mst::Bool = false, # compute min spannig tree
              kmst_out::String="",       # plot Kruskal min spann tree
              pmst_out::String="",       # plot Prim min spann tree
              graph_out::String="",      # plot the graph instance
              sol_out::String="")        # plot the solution tree

    # Generate the instance (or read it from a file)
    # Use graph_seed to fix the random instance (graph+weights)
    if isa(graph, Symbol)
        G = FactorGraph(N, Δ, α; graph_type=graph, graph_seed=graph_seed, c=c, σ=σ, root_id=root_id, init=mess_init)
    elseif isa(graph, String)
        G = FactorGraph(Δ, α, graph; root_id=root_id, init=mess_init)
        input_graph = readdlm(graph)
        N = Int(maximum(input_graph[:,1:2]))
    end
    seed > 0 && Random.seed!(seed)

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

    correct_steps = 0
    for t = 1:maxiter
        oneBPstep!(G; ρ=ρ)
        assign_variables!(G)
        is_good(G) ? (correct_steps += 1) : (correct_steps = 0)

        s = @sprintf("iter=%i, correct_steps=%i", t, correct_steps)
        ρ > 0.0 && (s *= @sprintf(", ρ=%3.3f ", ρ))
        print("\r$s")

        correct_steps == tconv && break
        ρ += ρstep
    end

    E = cost(G)
    correct_steps == tconv && @info "Final cost = $E"

    # Plot the solution tree
    if !isempty(sol_out)
        plot_sol_tree(G, sol_out)
    end

    terminal_nodes = [G.vnodes[i].is_terminal for i = 1:N]
    p = [G.vnodes[i].p for i = 1:N]
    d = [G.vnodes[i].d for i = 1:N]

    return E, terminal_nodes, p, d
end

end # module
