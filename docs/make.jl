using Documenter
using PencilFFTs
using Literate
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(
    PencilFFTs, :DocTestSetup, :(using PencilFFTs);
    recursive = false,
)

DocMeta.setdocmeta!(
    PencilFFTs.Transforms, :DocTestSetup, :(using PencilFFTs.Transforms);
    recursive = false,
)

literate_examples = [
    joinpath(@__DIR__, "examples", "gradient.jl"),
    joinpath(@__DIR__, "examples", "navier_stokes.jl"),
    joinpath(@__DIR__, "examples", "in-place.jl"),
]

const gendir = joinpath(@__DIR__, "src", "generated")

if rank == 0
    mkpath(gendir)
    examples = map(literate_examples) do inputfile
        outfile = Literate.markdown(inputfile, gendir)
        relpath(outfile, joinpath(@__DIR__, "src"))
    end
else
    examples = nothing
end

examples = MPI.bcast(examples, 0, comm) :: Vector{String}

@info "Example files (rank $rank): $examples"

@time makedocs(
    modules = [PencilFFTs],
    authors = "Juan Ignacio Polanco <jipolanc@gmail.com> and contributors",
    repo = "https://github.com/jipolanco/PencilFFTs.jl/blob/{commit}{path}#L{line}",
    sitename = "PencilFFTs.jl",
    format = Documenter.HTML(
        prettyurls = true,  # needed for correct path to movies (Navier-Stokes example)
        canonical = "https://jipolanco.github.io/PencilFFTs.jl",
        # load assets in <head>
        assets = [
            "assets/custom.css",
            "assets/tomate.js",
        ],
        mathengine = KaTeX(),
    ),
    build = rank == 0 ? "build" : mktempdir(),
    pages = [
        "Home" => "index.md",
        "tutorial.md",
        "Examples" => examples,
        "Library" => [
            "PencilFFTs.md",
            "Transforms.md",
            "PencilFFTs_timers.md",
            "Internals" => ["GlobalFFTParams.md"],
        ],
        "benchmarks.md",
    ],
    doctest = true,
    # linkcheck = true,
    linkcheck_ignore = [
        # This URL is correct, but gets incorrectly flagged by linkcheck.
        "https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.Pencils.range_local-Tuple{Union{PencilArray,%20Union{Tuple{Vararg{A,N}%20where%20N},%20AbstractArray{A,N}%20where%20N}%20where%20A%3C:PencilArray}}",
    ],
    # checkdocs = :all,
)

if rank == 0
    deploydocs(
        repo = "github.com/jipolanco/PencilFFTs.jl",
        forcepush = true,
        # PRs deploy at https://jipolanco.github.io/PencilFFTs.jl/previews/PR**
        push_preview = true,
    )
end
