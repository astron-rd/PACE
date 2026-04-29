module Util

using Printf

function print_header(message)
    println(repeat("=", 50))
    println(message)
    println(repeat("=", 50))
end

function time_function(name, func)
    t = @elapsed result = func()
    @printf("%-38s %10fs\n", name, t)
    result
end

function array_mod(x, mod)
    ((x - 1) % mod) + 1
end

end
