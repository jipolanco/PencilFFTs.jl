#!/usr/bin/env julia

# Based on
# https://gitlab.com/tkpapp/GitlabJuliaDemo.jl/blob/master/test/coverage/coverage-summary.jl

####
#### Coverage summary, printed as "(percentage) covered".
####
#### Useful for CI environments that just want a summary (eg a Gitlab setup).
####

using Coverage

cd(joinpath(@__DIR__, "..", "..")) do
    processed = process_folder("src")
    Codecov.submit_local(processed)
    covered_lines, total_lines = get_summary(processed)
    percentage = covered_lines / total_lines * 100
    println("($(percentage)%) covered")
end
