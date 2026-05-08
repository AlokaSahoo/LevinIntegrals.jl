using LevinIntegrals
using Documenter
using DocumenterVitepress

# DocMeta.setdocmeta!(LevinIntegrals, :DocTestSetup, :(using LevinIntegrals); recursive=true)


makedocs(;
    modules=[LevinIntegrals],
    authors="Aloka Kumar Sahoo <aloka_s@ph.iitr.ac.in>",
    sitename="LevinIntegrals.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo="github.com/AlokaSahoo/LevinIntegrals.jl",
        devbranch = "main", # or master, trunk, ...
        devurl = "dev",
    ),
    pages=[
        "Home" => "index.md"
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/AlokaSahoo/LevinIntegrals.jl",
    target = joinpath(@__DIR__, "build"),
    devbranch = "main",
    branch = "gh-pages",
    push_preview = true,
)
