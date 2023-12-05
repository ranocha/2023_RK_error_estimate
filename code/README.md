# Numerical experiments

This directory contains code to reproduce the numerical experiments described
the the manuscript. The code was developed with Julia v1.9.2. To reproduce the
results, start Julia in this directory and execute the following commands in
the Julia REPL.

Note that you can start Julia with several threads using, e.g.,
`julia --threads=4`. This is recommended for the last two PDE examples in
several space dimensions.

```julia
julia> include("code.jl")

julia> plot_step_size_control_stability_rk22()

julia> test(; alg = Heun(), estimate = :residual_quadratic_l1, betas = (1.0, 0.0))

julia> test(; alg = Heun(), estimate = :residual_quadratic_l2, betas = (1.0, 0.0))

julia> plot_step_size_control_stability_rk22_pi()

julia> test(; alg = Heun(), estimate = :residual_quadratic_l1, betas = (0.6, -0.2))

julia> test(; alg = Heun(), estimate = :residual_quadratic_l2, betas = (0.6, -0.2))

julia> plot_step_size_control_stability_bs3()

julia> plot_step_size_control_stability_bs3_pi()

julia> plot_step_size_control_stability_bs3_quadrature()

julia> test(; alg = BS3(), estimate = :residual_cubic_l1, betas = (1.0, 0.0))

julia> plot_step_size_control_stability_bs3_quadrature_pi()

julia> test(; alg = BS3(), estimate = :residual_cubic_l1, betas = (0.6, -0.2))

julia> plot_step_size_control_stability_bs3_l2()

julia> test(; alg = BS3(), estimate = :residual_cubic_l2, betas = (1.0, 0.0))

julia> plot_step_size_control_stability_bs3_l2_pi()

julia> test(; alg = BS3(), estimate = :residual_cubic_l2, betas = (0.6, -0.2))

julia> plot_krogh()

julia> plot_rigidbody()

julia> plot_bbm() # # this can take a few minutes

julia> plot_linear_advection() # this can take several minutes with multiple threads

julia> plot_compressible_euler() # this can take several minutes with multiple threads

```

This will print some information discussed in the paper to the terminal and
create the figures shown in the paper.

This directory also contains two Mathematica notebooks -
`step_size_control_with_L1.nb` and `step_size_control_with_L2.nb` -
containing analytical derivations/verifications of material presented in
the manuscript. These notebooks have been executed with Mathematica 12.1.
