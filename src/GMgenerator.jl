module GMgenerator

using Statistics, Random
using Plots
using DelimitedFiles

const F = Float64
const VF = Vector{F}
const VVF = Vector{VF}
const MF = Matrix{F}
const VI = Vector{Int}

# K centers are generated uniformly in [0,1]ᴰ, or at specified centers m(k)
# Nk (default N/K) points are generated as Normal(m(k), s(k)) for each k=1:K
# TODO:
# *) I want them to be norm in a [0,1]^D box
# *) Nk, m, s should overwrite N, D, K as long as they are consistent
function gen_points(N::Int, D::Int, K::Int;
                    seed::Int = -1,
                    Nk::Union{Nothing, VI}=nothing,
                    m::Union{Nothing, VVF}=nothing,
                    s::Union{F, VF}=0.1,
                    norm::Bool=true)

    (N < 2 && isnothing(Nk)) && throw(ArgumentError("N should be greater than one, got N=$N"))
    D > 1 || throw(ArgumentError("Dimension should be greater than one, got D=$D"))

    seed > 0 && Random.seed!(seed)

    # Centroids means
    if isnothing(m)
        m = [rand(D) for i = 1:K]
    else
        @assert length(m) == K
        for k = 1:K
            @assert length(m[k]) == D
        end
    end

    # Centroids variances
    if typeof(s) == F
        s = [s for k = 1:K] # I dont like this
    else
        @assert length(s) == K
    end

    # Number of points in each cluster
    if isnothing(Nk)
        Nk = div(N, K)
        if N - K * Nk > 0
            @warn "N=$N is not divisible by $K, Im going to generate $(K*Nk) points"
            N = K * Nk
        end
        Nk = [Nk for k = 1:K]
    else
        @assert length(Nk) == K
        newN = sum(Nk)
        if newN != N
            @warn "Im going to set N=sum(Nk)=$newN"
            N = newN
        end
    end

    # Responsibilities
    r = zeros(Int, N)
    idx = 1
    for k = 1:K
        for i = 1:Nk[k]
            r[idx] = k
            idx += 1
        end
    end

    data = hcat([m[k] .+ s[k] .* randn(D, Nk[k]) for k = 1:K]...)

    if norm
        M = maximum(abs.(data))
        data ./= M
        m ./= M
    end

    return data, m, r
end

function plot_points(data::MF;
                     m::Union{Nothing, VVF}=nothing,
                     r::Union{Nothing, VI}=nothing,
                     outfile::String="")

    D, N = size(data)
    @assert size(data)[1] == 2

    if isnothing(r)
        r = ones(N)
    else
        @assert length(r) == N
        K = maximum(r)
    end

    gr() # plotting backend

    p = scatter(data[1, r.==1], data[2, r.==1], legend=false)
    !isnothing(m) && scatter!(m[1][1, :], m[1][2, :], legend=false, marker=(8,:star5, :black))
    for k = 2:K
        scatter!(data[1, r.==k], data[2, r.==k], legend=false)
        !isnothing(m) && scatter!(m[k][1, :], m[k][2, :], legend=false, marker=(8,:star5, :black))
    end

    !isempty(outfile) && (png(p, outfile))
    display(p) # not needed in repl
end

function distance(a::AbstractArray, b::AbstractArray)
    d = length(a)
    @assert length(b) == d
    dist = 0.0
    for i = 1:d
        dist += (a[i] - b[i])^2
    end
    return sqrt(dist)
end

function similarity(data::MF;
                    λ::F=0.0,
                    use_mean::Bool=false,
                    use_median::Bool=false)
    D, N = size(data)
    S = zeros(F, N, N)

    for i = 1:N
        for j = (i+1):N
            d = @views distance(data[:,i], data[:,j])
            S[i,j] = d
            S[j,i] = d
        end
        λ > 0.0 && (S[i,i] = λ)
    end

    if use_mean
        λ = mean(S[S.!=0.0])
        for i = 1:N
            S[i,i] = λ
        end
    elseif use_median
        λ = median(S[S.!=0.0])
        for i = 1:N
            S[i,i] = λ
        end
    end

    return S
end

# takes as input the similarity matrix and
# writes a file in a form suitable for SteinerTreeFocusingBP analysis,
# i.e. i j wij with i < j
# if the selfaffinity is > 0, a node is added that acts as the root
# TODO:
# *) how can i avoid writing / reading?
# *) how do i communicate who is the root?
function write_graph_file(S::MF, outfile::String; tol::F=0.0)
    N = size(S)[1]
    @assert size(S)[2] == N

    rows = []
    if S[1,1] > 0.0
        λ = S[1,1]
        for i = 1:N
            push!(rows, [1 i+1 λ]) # node 1 is the root
        end
    end

    for i = 1:N
        for j = (i+1):N
            push!(rows, [i+1 j+1 S[i,j]])
        end
    end
    writedlm(outfile, rows)
end


end # module
