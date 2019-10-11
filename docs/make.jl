using Documenter
using PencilFFTs

makedocs(
    sitename = "PencilFFTs",
    format = Documenter.HTML(),
    modules = [PencilFFTs]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
