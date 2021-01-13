using Documenter
using PencilFFTs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(
    PencilFFTs, :DocTestSetup, :(using PencilFFTs); recursive=true)

function main()
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
                "assets/tomate.js",  # matomo code
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
        linkcheck=true,
        linkcheck_ignore=[
            # This URL is correct, but gets incorrectly flagged by linkcheck.
            "https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.Pencils.range_local-Tuple{Union{PencilArray,%20Union{Tuple{Vararg{A,N}%20where%20N},%20AbstractArray{A,N}%20where%20N}%20where%20A%3C:PencilArray}}",
        ],
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
