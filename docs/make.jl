using Documenter
using PencilFFTs

# This is to make sure that doctests in docstrings are executed correctly.
DocMeta.setdocmeta!(PencilFFTs, :DocTestSetup,
                    :(using PencilFFTs); recursive=false)
DocMeta.setdocmeta!(PencilFFTs.Pencils, :DocTestSetup,
                    :(using PencilFFTs.Pencils); recursive=true)
DocMeta.setdocmeta!(PencilFFTs.Transforms, :DocTestSetup,
                    :(using PencilFFTs.Transforms); recursive=true)

makedocs(
    sitename = "PencilFFTs",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [PencilFFTs],
    doctest = true,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
