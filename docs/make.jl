using Documenter
using HDF5  # to load HDF5 code via Requires
using PencilFFTs

const MAKE_FAST = Ref("--fast" in ARGS)  # skip some checks in makedocs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(PencilFFTs, :DocTestSetup,
                    :(using PencilFFTs); recursive=false)
DocMeta.setdocmeta!(PencilFFTs.PencilArrays, :DocTestSetup,
                    :(using PencilFFTs.PencilArrays); recursive=true)
DocMeta.setdocmeta!(PencilFFTs.Permutations, :DocTestSetup,
                    :(using PencilFFTs.Permutations); recursive=true)
DocMeta.setdocmeta!(PencilFFTs.Transforms, :DocTestSetup,
                    :(using PencilFFTs.Transforms); recursive=true)
DocMeta.setdocmeta!(PencilFFTs.PencilIO, :DocTestSetup,
                    :(using PencilFFTs.PencilIO); recursive=true)

function main()
    with_checks = !MAKE_FAST[]

    @time makedocs(
        sitename="PencilFFTs.jl",
        format=Documenter.HTML(
            prettyurls=true,
            # load assets in <head>
            assets=[
                "assets/custom.css",
                "assets/matomo.js",
            ],
        ),
        modules=[PencilFFTs],
        pages=[
            "Home" => "index.md",
            "tutorial.md",
            "More examples" => [
                "examples/in-place.md",
                "examples/gradient.md",
            ],
            "Library" => [
                "MPI-distributed arrays" => [
                    "MPITopology.md",
                    "Pencils.md",
                    "PencilArrays.md",
                    "Transpositions.md",
                    "PencilArrays_timers.md",
                    "Internals" => ["PermutationUtils.md"]
                ],
                "PencilIO.md",
                "Distributed FFTs" => [
                    "PencilFFTs.md",
                    "Transforms.md",
                    "PencilFFTs_timers.md",
                    "Internals" => ["GlobalFFTParams.md"],
                ]
            ],
            "benchmarks.md",
        ],
        doctest=true,
        linkcheck=with_checks,
        checkdocs=:all,
    )

    nothing
end

main()

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/jipolanco/PencilFFTs.jl",
    forcepush=true,
)
