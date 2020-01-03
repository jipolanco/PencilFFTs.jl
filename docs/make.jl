using Documenter
using PencilFFTs

const MAKE_FAST = "--fast" in ARGS  # skip some checks in makedocs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(PencilFFTs, :DocTestSetup,
                    :(using PencilFFTs); recursive=false)
DocMeta.setdocmeta!(PencilFFTs.Pencils, :DocTestSetup,
                    :(using PencilFFTs.Pencils); recursive=true)
DocMeta.setdocmeta!(PencilFFTs.Transforms, :DocTestSetup,
                    :(using PencilFFTs.Transforms); recursive=true)

let with_checks = !MAKE_FAST
    @time makedocs(
        sitename="PencilFFTs.jl",
        format=Documenter.HTML(
            prettyurls=true,
            # load assets in <head>
            assets=["assets/custom.css",
                    "assets/matomo.js"],
        ),
        modules=[PencilFFTs],
        pages=[
            "Home" => "index.md",
            "tutorial.md",
            "More examples" => [
                # "examples/init_vector_field.md",
                "examples/gradient.md",
            ],
            "Library" => [
                "PencilFFTs.md",
                "Transforms.md",
                "Pencils.md",
            ],
            "benchmarks.md",
        ],
        doctest=with_checks,
        linkcheck=with_checks,
        checkdocs=:all,
    )
end

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/jipolanco/PencilFFTs.jl",
    forcepush=true,
)
