module runexp

using Printf, DelimitedFiles
using Random, Statistics

include("../../src/SteinerTreeFocusingBP.jl"); S=SteinerTreeFocusingBP;

function main(N::Int;
              Δ_range::Vector{Int} = [2],
              α_range::Vector{Float64} = [0.5],
              nsamples::Int = 1,
              seed::Int = 0,
              graph_seed::Int = 0,
              graph::Union{Symbol, String} = :fc,
              root_id::Int = 1,
              c::Int = 3,
              σ::Float64 = 1.0,
              distr::Symbol = :unif,
              mess_init::Float64 = 0.0,
              maxiter::Int = 100,
              tconv::Int = 10,
              ρ::Float64 = 0.0,
              ρstep::Float64 = 0.0,
              verbose::Bool = true,
              verbose_algo::Bool = true,
              outfile::String = "",
              outfile_w::String = "")


    if length(Δ_range) > 1 && length(α_range) > 1
        error("Please choose to iterare over α or Δ")
    end

    if !isempty(outfile)
        f = open(outfile, "w")
    end

    for Δ in Δ_range
        for α in α_range
            ener = []
            stein = []
            conv = []
            solw = []
            for n = 1:nsamples
                # TODO: find a better way
                graph_seed += 1
                sol = S.main(N, Δ, α;
                            seed=seed, graph_seed=graph_seed,
                            graph=graph, distr=distr, c=c, root_id=root_id,
                            maxiter=maxiter, tconv=tconv, ρ=ρ, ρstep=ρstep,
                            verbose=verbose_algo);
                push!(ener, sol[1])
                push!(stein, sol[2])
                push!(conv, sol[3])
                push!(solw, sol[4]...)
            end
            if !isempty(outfile_w)
                writedlm(outfile_w * "_D$Δ" * "_a$α.dat", solw)
            end

            mean_ener = mean(ener)
            mean_stein = mean(stein)
            mean_conv = mean(conv)
            if nsamples > 1
                err_ener = std(ener) / sqrt(nsamples-1.0)
                err_stein = std(stein) / sqrt(nsamples-1.0)
                err_conv = std(conv) / sqrt(nsamples-1.0)
                out = @sprintf("Δ=%i, α=%.2f, e=%.4f±%.4f, s=%.4f±%.4f, conv=%.3f±%.3f",
                                Δ, α, mean_ener, err_ener, mean_stein, err_stein, mean_conv, err_conv)
                if !isempty(outfile)
                    println(f, "$Δ $α $mean_ener $err_ener $mean_stein $err_stein $mean_conv $err_conv")
                end
            else
                out = @sprintf("Δ=%i, α=%.2f, e=%.4f, s=%i, conv=%i",
                                Δ, α, mean_ener, mean_stein, mean_conv)
                if !isempty(outfile)
                    println(f, "$Δ $α $mean_ener $mean_stein $mean_conv")
                end
            end
            verbose && print("\n$out")
        end
    end

    if !isempty(outfile)
        close(f)
    end

end # main

end # module
