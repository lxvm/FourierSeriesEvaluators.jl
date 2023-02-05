push!(LOAD_PATH, "../src")
using Documenter, FourierSeriesEvaluators

Documenter.HTML(
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    )),
)

makedocs(
    sitename="FourierSeriesEvaluators.jl",
    modules=[FourierSeriesEvaluators],
    pages = [
        "Home" => "index.md",
        "Manual" => "methods.md",
    ],
)

deploydocs(
    repo = "github.com/lxvm/FourierSeriesEvaluators.jl.git",
)