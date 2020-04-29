#module Utils

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
    flag = false
    for (i, v) in enumerate(vnodes)
        for (j, nn) in enumerate(v.neighs)
            v.w[j] > 1.0 && (flag = true)
            distmx[i,nn] = v.w[j]
        end
    end
    flag && @warn "You have weights > 1, mst cost will be wrong"

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


#end # module
