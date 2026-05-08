using LevinIntegrals
using Documenter

DocMeta.setdocmeta!(LevinIntegrals, :DocTestSetup, :(using LevinIntegrals); recursive=true)

makedocs(;
    modules=[LevinIntegrals],
    authors="Aloka Kumar Sahoo",
    sitename="LevinIntegrals.jl",
    doctest=true,
    format=Documenter.HTML(;
        repolink=""https://github.com/AlokaSahoo/LevinIntegrals.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/AlokaSahoo/LevinIntegrals.jl",
    devbranch="main",
)
