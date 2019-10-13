using Documenter
using PencilFFTs

# This is to make sure that doctests in docstrings are executed correctly.
doctest_meta = quote
    using PencilFFTs
    using PencilFFTs.Pencils
end

DocMeta.setdocmeta!(PencilFFTs.Pencils, :DocTestSetup,
                    :(using PencilFFTs.Pencils); recursive=true)

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
