module runexp

include("../../src/FocusingAP.jl"); FAP = FocusingAP;
include("../../src/GMgenerator.jl"); G = GMgenerator;

using Statistics, Random
using DelimitedFiles, Printf
using Clustering
using NearestNeighbors

const F  = Float64
const MF = Matrix{F}
const VI = Vector{Int}

insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), [x]); v)

function get_assignments(S::DenseMatrix, exemplars::Vector{Int})
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

function compute_energy(S::Matrix{Float64}, exemplars::Vector{Int}, assignments::Vector{Int})
    E = 0
    N = size(S,2)

    for i in 1:N
        E -= S[i,exemplars[assignments[i]]]
    end

    return E
end

function explore(data::Union{MF, String};
                 true_r::Union{VI, String}=VI(),
                 skipstart_r::Int=4, # skip first skipstart_r lines in reading true_r file
                 norm::Bool=true,
                 #
                 nsamples::Int=1,
                 seed::Int=-1,
                 maxiter::Int=1000,
                 init::String="unif",
                 α::F=0.0,
                 damp::F=0.5,
                 #
                 λmin::F=-1.0,
                 λmax::F=-1.0,
                 add_meanS::Bool=false,
                 add_medianS::Bool=false,
                 λsteps::Int=50,
                 #
                 y::Float64=0.0,
                 yfact::Float64=0.0,
                 γ::Float64=0.0,
                 γfact::Float64=0.0,
                 #
                 kmax::Int=0, # if > 0, perturb the sol with k-th nearest exemplars
                 #
                 verbose::Bool=true,
                 outfile::String="",
                 outfile_histo::String="",
                 outfile_pert::String="",
                 outfile_count::String="")

    nsamples > 1 && @assert α > 0.0

    if isa(data, String)
        data = MF(readdlm(data)')
        norm && (data ./= maximum(data))
    end

    if isa(true_r, String)
        true_r = readdlm(true_r, skipstart=skipstart_r, Int)
        @assert length(true_r) == size(data)[2]
    else
        @assert length(true_r) == size(data)[2]
    end

    S = G.similarity(data)
    N = size(S)[2]


    minS, maxS = minimum(S[S.!=0.0]), maximum(S[S.!=0.0])
    meanS, medianS = mean(S[S.!=0.0]), median(S[S.!=0.0])
    λmin < 0.0 && (λmin = minS)
    λmax < 0.0 && (λmax = maxS)
    add_meanS   && insert_and_dedup!(λrange, meanS)
    add_medianS && insert_and_dedup!(λrange, medianS)

    λrange = [LinRange(λmin, λmax, λsteps)...]

    out = @sprintf("minS=%.4E, maxS=%.4E, <S>=%.4E, S̅=%.4E",
                    minS, maxS, meanS, medianS)
    verbose && @info out


    mean_iters = []; err_iters = []
    mean_pconvs = []; err_pconvs = []
    mean_numexs = []; err_numexs = []
    mean_eners = []; err_eners = []
    mean_accs = []; err_accs = []

    S .*=-1.0

    for λ in λrange

        if !isempty(outfile_count)
            excount = zeros(Int, N)
        end

        iter = zeros(Int, nsamples)
        pconv  = zeros(Int, nsamples)
        numex  = zeros(Int, nsamples)
        ener = zeros(F, nsamples)
        acc  = zeros(F, nsamples)

        pert_ener = zeros(F, nsamples, kmax)
        pert_acc  = zeros(F, nsamples, kmax)

        for n = 1:nsamples
            seedap = seed + n
            res = FAP.main(S; λ=-λ, seed=seedap, maxiter=maxiter, init=init, α=α, damp=damp,
                              γ=γ, γfact=γfact, y=y, yfact=yfact,
                              print=false, print_res=false);

            iter[n] = res.iterations
            pconv[n] = Int(res.converged)
            numex[n] = length(res.exemplars)
            ener[n] = res.energy
            acc[n] = randindex(res.assignments, true_r)[2]
            #
            # RI = randindex(r, true_r) # sort of accuracy
            # 1) Hubert & Arabie Adjusted Rand index
            # 2) Rand index (agreement probability)
            # 3) Mirkin's index (disagreement probability)
            # 4) Hubert's index (P(agree)−P(disagree))
            # V = varinfo(r, true_r) # sort of mi but is a metric
            # MI = mutualinfo(r, true_r)
            if !isempty(outfile_count)
                excount[res.exemplars] .+= 1
            end
            # Perturb solution with k-th nearest exemplars
            if kmax > 1
                pert_ener[n,:], pert_acc[n,:] = perturb_solution(data, S, res, true_r, kmax; verbose=verbose)
            end

            out = @sprintf("sample=%i", n)
            verbose && print("\r$out")

        end # for n

        @show size(pert_ener) size(pert_acc)

        # Write obs histograms over nsamples
        if nsamples > 1 && !isempty(outfile_histo)
            outfile_histo = "histo_" * outfile_histo
            outfile_histo *= "_l$λ" * "_" * init * "_a$α" * "_damp$damp"
            if γ > 0.0 || y > 0.0
                outfile_histo *= "_g$γ" * "_dg$γfact" * "_y$y" * "_dy$yfact"
            end
            outfile_histo *= "_n$nsamples" * "_seed$seed.dat"
            # iter | pconv | numex | ener | acc |
            writedlm(outfile_histo, [iter pconv numex ener acc])
        end

        # Write exemplars count over nsamples
        if nsamples > 1 && !isempty(outfile_count)
            outfile_count = "excount_" * outfile_count
            outfile_count *= "_l$λ" * "_" * init * "_a$α" * "_damp$damp"
            if γ > 0.0 || y > 0.0
                outfile_count *= "_g$γ" * "_dg$γfact" * "_y$y" * "_dy$yfact"
            end
            outfile_count *= "_n$nsamples" * "_seed$seed.dat"
            # iter | pconv | numex | ener | acc |
            writedlm(outfile_count, [excount])
        end

        # Write mean obs perturbations over nsamples
        if !isempty(outfile_pert)
            outfile_pert = "pert_" * outfile_pert
            outfile_pert *= "_k$kmax" * "_l$λ" * "_" * init * "_a$α" * "_damp$damp"
            if γ > 0.0 || y > 0.0
                outfile_pert *= "_g$γ" * "_dg$γfact" * "_y$y" * "_dy$yfact"
            end
            if nsamples > 1
                outfile_pert *= "_n$nsamples" * "_seed$seed.dat"
                norm = 1/sqrt(nsamples-1)
                # k | <Δener> | ± | <Δacc> | ± |
                writedlm(outfile_pert, [1:kmax mean(pert_ener; dims=1)' std(pert_ener; dims=1)'.*norm mean(pert_acc; dims=1)' std(pert_acc; dims=1)'.*norm])
            else
                outfile_pert *= "_seed$seed.dat"
                # k | <Δener> | <Δacc> |
                writedlm(outfile_pert, [1:kmax mean(pert_ener; dims=1)' mean(pert_acc; dims=1)'])
            end
        end
        #
        mean_iter = mean(iter)
        mean_pconv = mean(pconv)
        mean_numex = mean(numex)
        mean_ener = mean(ener)
        mean_acc = mean(acc)
        push!(mean_iters, mean_iter)
        push!(mean_pconvs, mean_pconv)
        push!(mean_numexs, mean_numex)
        push!(mean_eners, mean_ener)
        push!(mean_accs, mean_acc)

        if nsamples > 1
            norm = 1/sqrt(nsamples-1)
            err_iter = std(iter) * norm
            err_pconv = std(pconv) * norm
            err_numex = std(numex) * norm
            err_ener = std(ener) * norm
            err_acc = std(acc) * norm
            push!(err_iters, err_iter)
            push!(err_pconvs, err_pconv)
            push!(err_numexs, err_numex)
            push!(err_eners, err_ener)
            push!(err_accs, err_acc)

            out = @sprintf("λ=%.3f, it=%.1f±%.1f, p=%.2f±%.2f, ex=%.1f±%.1f, ener=%.2f±%.2f, acc=%.3f±%.3f",
                            λ, mean_iter, err_iter, mean_pconv, err_pconv, mean_numex, err_numex, mean_ener, err_ener, mean_acc, err_acc)
            verbose && print("\n$out\n")
        else
            out = @sprintf("λ=%.3f, it=%.1f, p=%.2f, ex=%.1f, ener=%.2f, acc=%.3f",
                            λ, mean_iter, mean_pconv, mean_numex, mean_ener, mean_acc)
            verbose && print("\n$out\n")
        end

    end # for λ

    # Write mean obs over nsamples vs λ
    if !isempty(outfile)
        outfile *= "_lm$λmin" * "_lM$λmax" * "_" * init * "_a$α" * "_damp$damp"
        if γ > 0.0 || y > 0.0
            outfile *= "_g$γ" * "_dg$γfact" * "_y$y" * "_dy$yfact"
        end
        if nsamples > 1
            outfile *= "_n$nsamples" * "_seed$seed.dat"
            # param | it | ± | pconv | ± | numex | ± | ener | ± | acc | ± |
            writedlm(outfile, [λrange mean_iters err_iters mean_pconvs err_pconvs mean_numexs err_numexs mean_eners err_eners mean_accs err_accs])
        else
            outfile *= "_seed$seed.dat"
            # param | it | pconv | numex | ener | acc |
            writedlm(outfile, [λrange mean_iters mean_pconvs mean_numexs mean_eners mean_accs])
        end
    end

end

function perturb_solution(data::MF, S::MF, res, true_r::VI, kmax::Int;
                          verbose::Bool=true,
                          outfile::String=""
                          )

    N = size(data)[2]
    @assert length(true_r) == N

    acc0 = randindex(res.assignments, true_r)[2]
    ener0 = res.energy

    exemplars = res.exemplars
    pert_ener = zeros(length(exemplars), kmax)
    pert_acc = zeros(length(exemplars), kmax)

    kdtree = KDTree(data)
    for (n, ex) in enumerate(exemplars)
        new_ex = copy(exemplars)
        point, dists = knn(kdtree, data[:,ex], kmax, true)
        for k = 1:kmax
            new_ex[n] = point[k]
            assignments, counts = get_assignments(S, new_ex)
            pert_ener[n,k] = compute_energy(S, new_ex, assignments)
            pert_acc[n,k] = randindex(assignments, true_r)[2]
        end
    end

    if !isempty(outfile)
        writedlm(outfile, [1:kmax ener' acc'])
    end

    if verbose
        for k = 1:kmax
            out = @sprintf("k=%i", k-1)
            for n = 1:length(exemplars)
                out *= @sprintf(", ener(%i)=%.4f", n, pert_ener[n,k])
            end
            for n = 1:length(exemplars)
                out *= @sprintf(", acc(%i)=%.4f", n, pert_acc[n,k])
            end
            print("$out\n")
        end
    end

    return mean(pert_ener,dims=1), mean(pert_acc,dims=1)

end

end # module
