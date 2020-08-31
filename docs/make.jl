using Documenter
using PencilFFTs

const MAKE_FAST = Ref("--fast" in ARGS)  # skip some checks in makedocs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(PencilFFTs.Transforms, :DocTestSetup,
                    :(using PencilFFTs.Transforms); recursive=true)

function main()
    with_checks = !MAKE_FAST[]

    @time makedocs(
        modules = [PencilFFTs],
        authors = "Juan Ignacio Polanco <jipolanc@gmail.com> and contributors",
        repo = "https://github.com/jipolanco/PencilFFTs.jl/blob/{commit}{path}#L{line}",
        sitename = "PencilFFTs.jl",
        format = Documenter.HTML(
            prettyurls=true,
            canonical = "https://jipolanco.github.io/PencilFFTs.jl",
            # load assets in <head>
            assets=[
                "assets/custom.css",
                "assets/matomo.js",
            ],
        ),
        pages=[
            "Home" => "index.md",
            "tutorial.md",
            "More examples" => [
                "examples/in-place.md",
                "examples/gradient.md",
            ],
            "Library" => [
                "PencilFFTs.md",
                "Transforms.md",
                "PencilFFTs_timers.md",
                "Internals" => ["GlobalFFTParams.md"],
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
    repo = "github.com/jipolanco/PencilFFTs.jl",
    forcepush = true,
)
