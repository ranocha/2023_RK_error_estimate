# Setup and install required packages
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using OrdinaryDiffEq
using DiffEqCallbacks: SavingCallback, SavedValues

# To approximate the integrals required for the residuabl-based error estimator
using QuadGK: quadgk
using LinearAlgebra: I, norm, ldiv!

# To differentiate integrals for numeric-analytical studies
using ForwardDiff: ForwardDiff

# For PDE tests
using SummationByPartsOperators
using Trixi

# For step size control stability plots
using LinearAlgebra: eigvals
using SimpleNonlinearSolve

using Plots: Plots, plot, plot!, scatter, scatter!, savefig
using LaTeXStrings


# A simple test problem of Hairer, Nørsett, Wanner II, eq. (2.27)
function hairer_norsett_wanner!(du, u, p, t)
  si, co = sincos(t)
  du[1] = -2000 * ( co * u[1] + si * u[2] + 1)
  du[2] = -2000 * (-si * u[1] + co * u[2] + 1)
  return nothing
end
function setup_hairer_norsett_wanner()
  u0 = zeros(2)
  u0[1] = 1
  tspan = (0.0, 1.57)
  ode = ODEProblem(hairer_norsett_wanner!, u0, tspan)
  return ode
end


# Run a simple numerical experiment comparing our home-grown RK loop including
# step size control with the algorithms implemented in OrdinaryDiffEq.jl.
function test(; alg = BS3(), estimate = :residual_quadratic_l1, betas = nothing,
                ode = setup_hairer_norsett_wanner())

  if betas === nothing
    # betas = (0.6, -0.2) # PI42 controller
    betas = (1.0, 0.0)  # deadbeat I controller
  end
  abstol = 1.0e-4     # absolute tolerance
  reltol = 1.0e-4     # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = @time ode_solve(ode, coefficients;
                                 dt, abstol, reltol,
                                 controller = PIDController(betas...),
                                 estimate)
  @info "residual estimator finished" sol_residual.stats.nf sol_residual.stats.naccept sol_residual.stats.nreject

  # Solve the ODE using OrdinaryDiffEq.jl
  sol_embedded = @time solve(ode, alg;
                             dt, abstol, reltol,
                             controller = PIDController(betas...))
  @info "embedded estimator finished" sol_embedded.stats.nf sol_embedded.stats.naccept sol_embedded.stats.nreject

  difference = sol_residual.u[end] - sol_embedded.u[end]
  @info "difference" difference

  fig_sol = plot(xguide = L"Time $t$", yguide = L"u")
  plot!(fig_sol, sol_residual.t, first.(real.(sol_residual.u));
        label = L"$u_1$, residual")
  plot!(fig_sol, sol_embedded.t, first.(real.(sol_embedded.u));
        label = L"$u_1$, embedded", linestyle = :dot)

  fig_dt = plot(xguide = L"Time $t$", yguide = L"Time step size $\Delta t$")
  plot!(fig_dt, sol_residual.t[begin:(end-1)], diff(sol_residual.t);
        label = "residual", plot_kwargs()...)
  plot!(fig_dt, sol_embedded.t[begin:(end-1)], diff(sol_embedded.t);
        label = "embedded", linestyle = :dot, plot_kwargs()...)

  return plot(fig_sol, fig_dt)
end



function plot_step_size_control_stability_rk22()
  stability_polynomial(z) = 1 + z + z^2 / 2

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2) / (1 + z + z^2 / 2))
    jacobian = [1 u;
                -1/k (1 - 3 / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 2
    # R(z) = 1 + z + z^2 / 2
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2) / (1 + z + z^2 / 2))
    # E(z) = R(z) - Rhat(z) = z^2 / 2
    # v = Re( E'(z) * z / E(z))
    v = 2
    jacobian = [1 u;
                -1/k (1 - v / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 3); label = L"residual ($k = 3$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 2$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_rk22.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_rk22_pi(β1 = 0.6, β2 = -0.2)
  stability_polynomial(z) = 1 + z + z^2 / 2

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2) / (1 + z + z^2 / 2))
    jacobian = [1 u 0 0;
                -β1/k (1 - 3 * β1 / k) -β2/k -3*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 2
    # R(z) = 1 + z + z^2 / 2
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2) / (1 + z + z^2 / 2))
    # E(z) = R(z) - Rhat(z) = z^2 / 2
    # v = Re( E'(z) * z / E(z))
    v = 2
    jacobian = [1 u 0 0;
                -β1/k (1 - v * β1 / k) -β2/k -v*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 3); label = L"residual ($k = 3$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 2$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_rk22_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3()
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    jacobian = [1 u;
                -1/k (1 - 4 / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u;
                -1/k (1 - v / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3_pi(β1 = 0.6, β2 = -0.2)
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    jacobian = [1 u 0 0;
                -β1/k (1 - 4 * β1 / k) -β2/k -4*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u 0 0;
                -β1/k (1 - v * β1 / k) -β2/k -v*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :topright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3_l2()
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    J22 = 1 - (64 + 10 * abs2(z) - 45 * real(z)) / (2 * k * (8 + abs2(z) - 5 * real(z)))
    jacobian = [1 u;
                -1/k J22]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u;
                -1/k (1 - v / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3_l2.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3_l2_pi(β1 = 0.6, β2 = -0.2)
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    J22_diff = (64 + 10 * abs2(z) - 45 * real(z)) / (2 * k * (8 + abs2(z) - 5 * real(z)))
    jacobian = [1 u 0 0;
                -β1/k (1 - β1 * J22_diff) -β2/k -β2*J22_diff;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u 0 0;
                -β1/k (1 - v * β1 / k) -β2/k -v*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :topright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3_l2_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3_quadrature()
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function estimate(eta, chi, z)
    u0 = exp(eta)
    dt = exp(chi)
    real_z = real(z)
    imag_z = imag(z)
    real_λ = real_z / dt
    imag_λ = imag_z / dt

    # Central cubic Hermite interpolation
    function integrand(t, dt)
      return (1/6) * abs(u0) * t * (dt - t) * sqrt((t - 2 * dt + t * dt * real_λ)^2 + (t * dt * imag_λ)^2) * sqrt(real_λ^2 + imag_λ^2)^4
    end
    integral, err = quadgk(t -> integrand(t, dt), 0.0, dt; rtol = 1.0e-8, atol = 1.0e-10)

    return integral
  end

  function diff_estimate_chi(eta, chi, z)
    u0 = exp(eta)
    dt = exp(chi)
    real_z = real(z)
    imag_z = imag(z)
    real_λ = real_z / dt
    imag_λ = imag_z / dt

    # Central cubic Hermite interpolation
    function integrand(t, dt)
      return (1/6) * abs(u0) * t * (dt - t) * sqrt((t - 2 * dt + t * dt * real_λ)^2 + (t * dt * imag_λ)^2) * sqrt(real_λ^2 + imag_λ^2)^4
    end
    integrand_derivative(tt, dtt) = ForwardDiff.derivative(dt -> integrand(tt, dt), dtt)
    integral, err = quadgk(t -> integrand_derivative(t, dt), 0.0, dt; rtol = 1.0e-8, atol = 1.0e-14)

    return (integral + integrand(dt, dt)) * dt
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    J22 = 1 - diff_estimate_chi(1.0, 1.0, z) / (k * estimate(1.0, 1.0, z))
    jacobian = [1 u;
                -1/k J22]

    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u;
                -1/k (1 - v / k)]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3_quadrature.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_step_size_control_stability_bs3_quadrature_pi(β1 = 0.6, β2 = -0.2)
  stability_polynomial(z) = 1 + z + z^2 / 2 + z^3 / 6

  stability_function_minus_one(r, phi) = abs(stability_polynomial(r * exp(im * phi))) - 1

  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  function compute_z(phi)
    bounds = (0.1, 3.0)
    prob = IntervalNonlinearProblem(stability_function_minus_one, bounds, phi)
    sol = solve(prob, Brent())
    r = sol.u
    z = r * exp(im * phi)
    return z
  end

  function estimate(eta, chi, z)
    u0 = exp(eta)
    dt = exp(chi)
    real_z = real(z)
    imag_z = imag(z)
    real_λ = real_z / dt
    imag_λ = imag_z / dt

    # Central cubic Hermite interpolation
    function integrand(t, dt)
      return (1/6) * abs(u0) * t * (dt - t) * sqrt((t - 2 * dt + t * dt * real_λ)^2 + (t * dt * imag_λ)^2) * sqrt(real_λ^2 + imag_λ^2)^4
    end
    integral, err = quadgk(t -> integrand(t, dt), 0.0, dt; rtol = 1.0e-8, atol = 1.0e-10)

    return integral
  end

  function diff_estimate_chi(eta, chi, z)
    u0 = exp(eta)
    dt = exp(chi)
    real_z = real(z)
    imag_z = imag(z)
    real_λ = real_z / dt
    imag_λ = imag_z / dt

    # Central cubic Hermite interpolation
    function integrand(t, dt)
      return (1/6) * abs(u0) * t * (dt - t) * sqrt((t - 2 * dt + t * dt * real_λ)^2 + (t * dt * imag_λ)^2) * sqrt(real_λ^2 + imag_λ^2)^4
    end
    integrand_derivative(tt, dtt) = ForwardDiff.derivative(dt -> integrand(tt, dt), dtt)
    integral, err = quadgk(t -> integrand_derivative(t, dt), 0.0, dt; rtol = 1.0e-8, atol = 1.0e-14)

    return (integral + integrand(dt, dt)) * dt
  end

  function spectral_radius_residual(phi, k)
    z = compute_z(phi)
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    v = diff_estimate_chi(1.0, 1.0, z) / estimate(1.0, 1.0, z)
    jacobian = [1 u 0 0;
                -β1/k (1 - v * β1 / k) -β2/k -v*β2/k;
                1 0 0 0;
                0 1 0 0]

    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  function spectral_radius_embedded(phi)
    z = compute_z(phi)
    k = 3
    # R(z) = 1 + z + z^2 / 2 + z^3 / 6
    # u = Re( R'(z) * z / R(z))
    u = real((z + z^2 + z^3 / 2) / (1 + z + z^2 / 2 + z^3 / 6))
    # E(z) = R(z) - Rhat(z)
    #      = (1 + z + z^2 / 2 + z^3 / 6) - (1 + z + z^2 / 2 + 3/16 * z^3 + 1/48 * z^4)
    #      = -1/48 * z^3 - 1/48 * z^4
    # E'(z) * z / E(z) = (3 * z^3 + 4 * z^4) / (z^3 + z^4)
    # v = Re( E'(z) * z / E(z))
    v = real((3 * z^3 + 4 * z^4) / (z^3 + z^4))
    jacobian = [1 u 0 0;
                -β1/k (1 - v * β1 / k) -β2/k -v*β2/k;
                1 0 0 0;
                0 1 0 0]
    λ = eigvals(jacobian)
    return maximum(abs, λ)
  end

  phi = range(π / 2, π, length = 200)
  fig = plot(xguide = L"Argument $\varphi$", yguide = L"Spectral radius of $J$")
  plot!(fig, phi, spectral_radius_residual.(phi, 4); label = L"residual ($k = 4$)",
        plot_kwargs()...)
  plot!(fig, phi, spectral_radius_embedded.(phi); label = L"embedded ($k = 3$)",
        plot_kwargs()..., linestyle = :dot)
  plot!(fig, legend = :bottomright, xtick = pitick(phi[begin], phi[end], 8))
  savefig(fig, joinpath(figdir, "spectral_radius_bs3_quadrature_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end

function plot_kwargs()
  fontsizes = (
    xtickfontsize = 14, ytickfontsize = 14,
    xguidefontsize = 16, yguidefontsize = 16,
    legendfontsize = 14)
  (; linewidth = 3, gridlinewidth = 2,
     markersize = 8, markerstrokewidth = 4,
     fontsizes...)
end

# Taken from https://discourse.julialang.org/t/plot-with-x-axis-in-radians-with-ticks-at-multiples-of-pi/65325/3
function pitick(start, stop, denom; mode=:text)
  a = Int(cld(start, π/denom))
  b = Int(fld(stop, π/denom))
  tick = range(a*π/denom, b*π/denom; step=π/denom)
  ticklabel = piticklabel.((a:b) .// denom, Val(mode))
  tick, ticklabel
end
function piticklabel(x::Rational, ::Val{:text})
  iszero(x) && return "0"
  S = x < 0 ? "-" : ""
  n, d = abs(numerator(x)), denominator(x)
  N = n == 1 ? "" : repr(n)
  d == 1 && return S * N * "π"
  S * N * "π/" * repr(d)
end
function piticklabel(x::Rational, ::Val{:latex})
  iszero(x) && return L"0"
  S = x < 0 ? "-" : ""
  n, d = abs(numerator(x)), denominator(x)
  N = n == 1 ? "" : repr(n)
  d == 1 && return L"%$S%$N\pi"
  L"%$S\frac{%$N\pi}{%$d}"
end



################################################################################
# Run a simple numerical experiment comparing our home-grown RK loop including
# step size control with the algorithms implemented in OrdinaryDiffEq.jl.
# This example tests the number of step rejections.

# A nonlinear test problem of Krogh (1973)
function krogh!(du, u, p, t)
  θ = p[1]
  U = fill(0.5, (4, 4))
  U[1, 1] = U[2, 2] = U[3, 3] = U[4, 4] = -0.5
  si, co = sincos(θ)
  B = [-10co -10si 0 0
       10si -10co 0 0
       0 0 1 0
       0 0 0 0.5]
  B = U' * B * U
  z = U * u
  du .= -B * u + U' * [0.5 * z[1]^2 - 0.5 * z[2]^2, z[1] * z[2], z[3]^2, z[4]^2]
  return nothing
end
function setup_krogh(θ)
  u0 = [0.0, -2.0, -1.0, -1.0]
  tspan = (0.0, 100.0)
  ode = ODEProblem(krogh!, u0, tspan, [θ])
  return ode
end

function solve_krogh(θ, alg, betas, estimate)
  ode = setup_krogh(θ)

  abstol = 1.0e-4     # absolute tolerance
  reltol = 1.0e-4     # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = ode_solve(ode, coefficients;
                           dt, abstol, reltol,
                           controller = PIDController(betas...),
                           estimate)

  # Solve the ODE using OrdinaryDiffEq.jl
  sol_embedded = solve(ode, alg;
                       dt, abstol, reltol,
                       controller = PIDController(betas...))

  return (; stats_residual = sol_residual.stats,
            stats_embedded = sol_embedded.stats,)
end

function plot_krogh(; estimate = :residual_cubic_l1)
  θ = range(π / 2, π, length = 100)
  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  # I controller
  alg = BS3()
  betas = (1.0, 0.0)
  accept_residual = zero(θ)
  reject_residual = zero(θ)
  accept_embedded = zero(θ)
  reject_embedded = zero(θ)
  for i in eachindex(θ)
    res = solve_krogh(θ[i], alg, betas, estimate)
    accept_residual[i] = res.stats_residual.naccept
    reject_residual[i] = res.stats_residual.nreject
    accept_embedded[i] = res.stats_embedded.naccept
    reject_embedded[i] = res.stats_embedded.nreject
  end

  fig = plot(xguide = L"Argument $\varphi$", yguide = "Number of rejected steps")
  scatter!(fig, θ, reject_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, θ, reject_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xtick = pitick(θ[begin], θ[end], 8))
  savefig(joinpath(figdir, "krogh_bs3_rejected_steps.pdf"))

  # PI controller
  alg = BS3()
  betas = (0.6, -0.2)
  accept_residual = zero(θ)
  reject_residual = zero(θ)
  accept_embedded = zero(θ)
  reject_embedded = zero(θ)
  for i in eachindex(θ)
    res = solve_krogh(θ[i], alg, betas, estimate)
    accept_residual[i] = res.stats_residual.naccept
    reject_residual[i] = res.stats_residual.nreject
    accept_embedded[i] = res.stats_embedded.naccept
    reject_embedded[i] = res.stats_embedded.nreject
  end

  fig = plot(xguide = L"Argument $\varphi$", yguide = "Number of rejected steps")
  scatter!(fig, θ, reject_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, θ, reject_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xtick = pitick(θ[begin], θ[end], 8))
  savefig(joinpath(figdir, "krogh_bs3_rejected_steps_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end



################################################################################
# Run a simple numerical experiment comparing our home-grown RK loop including
# step size control with the algorithms implemented in OrdinaryDiffEq.jl.
# This example tests the numerical error.

# The Euler equations of a rigid body. See also Krogh (1973)
# The solution is periodic with period 4K,
# K = 1.862640802332738552030281220579...
function euler!(du, u, p, t)
  du[1] = u[2] * u[3]
  du[2] = -u[1] * u[3]
  du[3] = -0.51 * u[1] * u[2]
  return nothing
end
function setup_rigidbody()
  u0 = [0.0, 1.0, 1.0]
  K = 1.862640802332738552030281220579
  tspan = (0.0, 4 * K)
  ode = ODEProblem(euler!, u0, tspan)
  return ode
end

function solve_rigidbody(alg, betas, estimate, tol)
  ode = setup_rigidbody()

  abstol = tol # absolute tolerance
  reltol = tol # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = ode_solve(ode, coefficients;
                           dt, abstol, reltol,
                           controller = PIDController(betas...),
                           estimate)

  # Solve the ODE using OrdinaryDiffEq.jl
  sol_embedded = solve(ode, alg;
                       dt, abstol, reltol,
                       controller = PIDController(betas...))

  error_residual = norm(sol_residual.u[end] - ode.u0)
  error_embedded = norm(sol_embedded.u[end] - ode.u0)

  return (; stats_residual = sol_residual.stats,
            stats_embedded = sol_embedded.stats,
            error_residual,
            error_embedded)
end

function plot_rigidbody(; estimate = :residual_cubic_l1)
  tol = 10.0 .^ range(-9, -2, length = 15)
  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  # PI controller
  alg = BS3()
  betas = (0.6, -0.2)
  accept_residual = zero(tol)
  reject_residual = zero(tol)
  error_residual = zero(tol)
  accept_embedded = zero(tol)
  reject_embedded = zero(tol)
  error_embedded = zero(tol)
  for i in eachindex(tol)
    res = solve_rigidbody(alg, betas, estimate, tol[i])
    accept_residual[i] = res.stats_residual.naccept
    reject_residual[i] = res.stats_residual.nreject
    error_residual[i] = res.error_residual
    accept_embedded[i] = res.stats_embedded.naccept
    reject_embedded[i] = res.stats_embedded.nreject
    error_embedded[i] = res.error_embedded
  end

  fig = plot(xguide = "Number of steps", yguide = "Error")
  scatter!(fig, accept_residual + reject_residual, error_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, accept_embedded + reject_embedded, error_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, yscale = :log10)
  savefig(joinpath(figdir, "rigidbody_bs3_work_precision_pi.pdf"))

  fig = plot(xguide = "Tolerance", yguide = "Error")
  scatter!(fig, tol, error_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tol, error_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, yscale = :log10, legend = :topleft)
  savefig(joinpath(figdir, "rigidbody_bs3_error_vs_tol_pi.pdf"))

  fig = plot(xguide = "Tolerance", yguide = "Number of rejected steps")
  scatter!(fig, tol, reject_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tol, reject_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, legend = :topleft)
  savefig(joinpath(figdir, "rigidbody_bs3_rejsteps_vs_tol_pi.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end



################################################################################
## BBM equation
function bbm_rhs!(du, u, param, t)
  (; D1, invImD2, tmp1, tmp2) = param
  one_third = one(t) / 3

  # this semidiscretization conserves the linear and quadratic invariants
  @. tmp1 = -one_third * u^2
  mul!(tmp2, D1, tmp1)
  mul!(tmp1, D1, u)
  @. tmp2 += -one_third * u * tmp1 - tmp1
  ldiv!(du, invImD2, tmp2)

  return nothing
end

function bbm_solution(t, x)
  # Physical setup of a traveling wave solution with speed `c`
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  A = 3 * (c - 1)
  K = 0.5 * sqrt(1 - 1 / c)
  x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

  return A / cosh(K * x_t)^2
end

function bbm_setup(; domain_traversals = 1)
  nnodes = 2^8
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  D1 = fourier_derivative_operator(xmin, xmax, nnodes)
  D2 = D1^2
  invImD2 = I - D2

  tspan = (0.0, (xmax - xmin) / (3 * c) + domain_traversals * (xmax - xmin) / c)
  x = grid(D1)
  u0 = bbm_solution.(tspan[1], x)
  tmp1 = similar(u0)
  tmp2 = similar(u0)
  param = (; D1, D2, invImD2, tmp1, tmp2)

  return ODEProblem(bbm_rhs!, u0, tspan, param)
end

function test_bbm(; alg = BS3(), betas = (0.6, -0.2),
                    estimate = :residual_cubic_l1,
                    estimate_tolerance = 1.0e-6,
                    tol = 1.0e-4,
                    plot_results = true)
  # Setup spatial semidiscretization
  ode = bbm_setup()


  # Time integration parameters
  abstol = tol  # absolute tolerance
  reltol = tol  # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = @time ode_solve(ode, coefficients;
                                 dt, abstol, reltol,
                                 controller = PIDController(betas...),
                                 estimate, estimate_tolerance,
                                 save_everystep = false)
  @info "residual estimator finished" sol_residual.stats.nf sol_residual.stats.naccept sol_residual.stats.nreject

  # Solve the ODE using OrdinaryDiffEq.jl
  saved_values = SavedValues(Float64, typeof(nothing))
  saving_callback = SavingCallback(Returns(nothing), saved_values)
  sol_embedded = @time solve(ode, alg;
                             callback = saving_callback,
                             dt, abstol, reltol,
                             controller = PIDController(betas...),
                             save_everystep = false)
  @info "embedded estimator finished" sol_embedded.stats.nf sol_embedded.stats.naccept sol_embedded.stats.nreject

  x = grid(ode.p.D1)
  @. ode.p.tmp1 = sol_residual.u[end] - bbm_solution(sol_residual.t[end], x)
  error_residual = integrate(abs2, ode.p.tmp1, ode.p.D1) |> sqrt
  @. ode.p.tmp1 = sol_embedded.u[end] - bbm_solution(sol_embedded.t[end], x)
  error_embedded = integrate(abs2, ode.p.tmp1, ode.p.D1) |> sqrt
  @info "errors" error_residual error_embedded

  if plot_results
    fig_dt = plot(xguide = L"Time $t$", yguide = L"Time step size $\Delta t$")
    plot!(fig_dt, sol_residual.t[begin:(end-1)], diff(sol_residual.t);
          label = "residual", plot_kwargs()...)
    plot!(fig_dt, saved_values.t[begin:(end-1)], diff(saved_values.t);
          label = "embedded", linestyle = :dot, plot_kwargs()...)
  else
    fig_dt = nothing
  end

  return (; fig_dt, error_residual, error_embedded, sol_residual, sol_embedded)
end

function plot_bbm(; estimate = :residual_cubic_l1)
  estimate_tolerance = 1.0e-6
  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  res = test_bbm(; tol = 1.0e-4, estimate, estimate_tolerance)
  savefig(res.fig_dt, joinpath(figdir, "bbm_dt_vs_t__BS3_cubicL1_tol1em4.pdf"))
  fig_sol = plot(xguide = L"x", yguide = L"u")
  plot!(fig_sol, grid(res.sol_embedded.prob.p.D1), res.sol_residual.u[end];
        label = "residual", plot_kwargs()...)
  plot!(fig_sol, grid(res.sol_embedded.prob.p.D1), res.sol_embedded.u[end];
        label = "embedded", linestyle = :dot, plot_kwargs()...)
  plot!(grid(res.sol_embedded.prob.p.D1), res.sol_embedded.u[begin];
        label = L"u^0", linestyle = :dashdot, color = :gray, plot_kwargs()...)
  savefig(fig_sol, joinpath(figdir, "bbm_solutions__BS3_cubicL1_tol1em4.pdf"))

  tolerances = 10.0 .^ range(-2.5, -6, length = 12)
  accept_residual = zero(tolerances)
  reject_residual = zero(tolerances)
  errors_residual = zero(tolerances)
  accept_embedded = zero(tolerances)
  reject_embedded = zero(tolerances)
  errors_embedded = zero(tolerances)

  for i in eachindex(tolerances, errors_embedded, errors_residual)
    res = test_bbm(; tol = tolerances[i], plot_results = false,
                     estimate, estimate_tolerance)
    accept_residual[i] = res.sol_residual.stats.naccept
    reject_residual[i] = res.sol_residual.stats.nreject
    errors_residual[i] = res.error_residual
    accept_embedded[i] = res.sol_embedded.stats.naccept
    reject_embedded[i] = res.sol_embedded.stats.nreject
    errors_embedded[i] = res.error_embedded
  end
  fig = plot(xguide = "Tolerance", yguide = L"$L^2$ error")
  scatter!(fig, tolerances, errors_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tolerances, errors_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, yscale = :log10, legend = :topleft)
  savefig(fig, joinpath(figdir, "bbm_error_vs_tol__BS3_cubicL1.pdf"))

  fig = plot(xguide = "Tolerance", yguide = "Number of rejected steps")
  scatter!(fig, tolerances, reject_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tolerances, reject_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, legend = :topleft)
  savefig(fig, joinpath(figdir, "bbm_rejsteps_vs_tol__BS3_cubicL1__NOT_USED.pdf"))

  fig = plot(xguide = "Number of steps", yguide = L"$L^2$ error")
  scatter!(fig, accept_residual + reject_residual, errors_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, accept_embedded + reject_embedded, errors_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, yscale = :log10, legend = :topright)
  savefig(fig, joinpath(figdir, "bbm_work_precision__BS3_cubicL1.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end



################################################################################
# Run a simple numerical experiment comparing our home-grown RK loop including
# step size control with the algorithms implemented in OrdinaryDiffEq.jl.
function test_linear_advection(; alg = BS3(), betas = (0.6, -0.2),
                                 estimate = :residual_cubic_l1,
                                 estimate_tolerance = 1.0e-8,
                                 tol = 1.0e-4,
                                 plot_results = true)
  # Setup spatial semidiscretization
  advection_velocity = (1.0, 1.0)
  equations = LinearScalarAdvectionEquation2D(advection_velocity)

  function initial_condition(x, t, equation::LinearScalarAdvectionEquation2D)
    x_trans = x - equation.advection_velocity * t
    c = 1.0
    A = 0.5
    L = 2
    f = 1 / L
    omega = 2 * π * f
    scalar = c + A * sin(omega * sum(x_trans))
    return SVector(scalar)
  end

  solver = DGSEM(polydeg = 4)

  coordinates_min = (-1.0, -1.0)
  coordinates_max = (+1.0, +1.0)
  mesh = TreeMesh(coordinates_min, coordinates_max;
                  initial_refinement_level = 3,
                  n_cells_max = 10_000,
                  periodicity = true)

  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                      solver)
  tspan = (0.0, 10.0)
  ode = semidiscretize(semi, tspan)

  analysis_callback = AnalysisCallback(semi)


  # Time integration parameters
  abstol = tol  # absolute tolerance
  reltol = tol  # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = @time ode_solve(ode, coefficients;
                                 dt, abstol, reltol,
                                 controller = PIDController(betas...),
                                 estimate, estimate_tolerance)
  @info "residual estimator finished" sol_residual.stats.nf sol_residual.stats.naccept sol_residual.stats.nreject

  # Solve the ODE using OrdinaryDiffEq.jl
  sol_embedded = @time solve(ode, alg;
                             dt, abstol, reltol,
                             controller = PIDController(betas...))
  @info "embedded estimator finished" sol_embedded.stats.nf sol_embedded.stats.naccept sol_embedded.stats.nreject

  errors_residual = analysis_callback(sol_residual)
  errors_embedded = analysis_callback(sol_embedded)
  @info "errors" errors_residual errors_embedded

  if plot_results
    fig_dt = plot(xguide = L"Time $t$", yguide = L"Time step size $\Delta t$")
    plot!(fig_dt, sol_residual.t[begin:(end-1)], diff(sol_residual.t);
          label = "residual", plot_kwargs()...)
    plot!(fig_dt, sol_embedded.t[begin:(end-1)], diff(sol_embedded.t);
          label = "embedded", linestyle = :dot, plot_kwargs()...)
  else
    fig_dt = nothing
  end

  return (; fig_dt, errors_residual, errors_embedded,
            sol_residual, sol_embedded)
end

function plot_linear_advection(; estimate = :residual_cubic_l1)
  estimate_tolerance = 1.0e-6
  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  res = test_linear_advection(; tol = 1.0e-4, estimate, estimate_tolerance)
  savefig(res.fig_dt, joinpath(figdir, "linadv_dt_vs_t__BS3_cubicL1_tol1em4.pdf"))

  tolerances = 10.0 .^ range(-2.5, -6, length = 12)
  accept_residual = zero(tolerances)
  reject_residual = zero(tolerances)
  errors_residual = zero(tolerances)
  accept_embedded = zero(tolerances)
  reject_embedded = zero(tolerances)
  errors_embedded = zero(tolerances)

  for i in eachindex(tolerances, errors_embedded, errors_residual)
    res = test_linear_advection(; tol = tolerances[i], plot_results = false,
                                  estimate, estimate_tolerance)
    accept_residual[i] = res.sol_residual.stats.naccept
    reject_residual[i] = res.sol_residual.stats.nreject
    errors_residual[i] = res.errors_residual.l2[1]
    accept_embedded[i] = res.sol_embedded.stats.naccept
    reject_embedded[i] = res.sol_embedded.stats.nreject
    errors_embedded[i] = res.errors_embedded.l2[1]
  end
  fig = plot(xguide = "Tolerance", yguide = L"$L^2$ error")
  scatter!(fig, tolerances, errors_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tolerances, errors_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10, yscale = :log10)
  savefig(fig, joinpath(figdir, "linadv_error_vs_tol__BS3_cubicL1.pdf"))

  fig = plot(xguide = "Tolerance", yguide = "Number of rejected steps")
  scatter!(fig, tolerances, reject_residual;
           label = "residual", marker = :x, plot_kwargs()...)
  scatter!(fig, tolerances, reject_embedded;
           label = "embedded", marker = :+, plot_kwargs()...)
  plot!(fig, xscale = :log10)
  savefig(fig, joinpath(figdir, "linadv_rejsteps_vs_tol__BS3_cubicL1.pdf"))

  # Very similar to the L1 estimate
  # for i in eachindex(tolerances, errors_embedded, errors_residual)
  #   res = test_linear_advection(; tol = tolerances[i], plot_results = false,
  #                                 estimate = :residual_cubic_l2,
  #                                 estimate_tolerance)
  #   errors_embedded[i] = res.errors_embedded.l2[1]
  #   errors_residual[i] = res.errors_residual.l2[1]
  # end
  # scatter(tolerances, errors_residual;
  #         label = "residual", marker = :x, plot_kwargs()...)
  # scatter!(tolerances, errors_embedded;
  #          label = "embedded", marker = :+, plot_kwargs()...)
  # plot!(xguide = "Tolerance", yguide = L"$L^2$ error",
  #       xscale = :log10, yscale = :log10)
  # savefig(joinpath(figdir, "linadv_error_vs_tol__BS3_cubicL2.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end



################################################################################
# Run a simple numerical experiment comparing our home-grown RK loop including
# step size control with the algorithms implemented in OrdinaryDiffEq.jl.
function test_compressible_euler(; alg = BS3(), betas = (0.6, -0.2),
                                 estimate = :residual_cubic_l1,
                                 estimate_tolerance = 1.0e-4,
                                 tol = 1.0e-5,
                                 plot_results = true)
  # Setup spatial semidiscretization
  equations = CompressibleEulerEquations3D(1.4)

  function initial_condition(x, t, equations::CompressibleEulerEquations3D)
    A  = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3  = 0.0
    p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p   = p + 1.0/16.0 * A^2 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
  end

  volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha_turbo)
  solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs;
                 volume_integral)

  coordinates_min = (-1.0, -1.0, -1.0) .* pi
  coordinates_max = ( 1.0,  1.0,  1.0) .* pi
  mesh = TreeMesh(coordinates_min, coordinates_max;
                  initial_refinement_level = 3,
                  n_cells_max = 10_000,
                  periodicity = true)

  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                      solver)
  tspan = (0.0, 10.0)
  ode = semidiscretize(semi, tspan)

  analysis_callback = AnalysisCallback(semi)


  # Time integration parameters
  abstol = tol  # absolute tolerance
  reltol = tol  # relative tolerance

  # the algorithms from OrdinaryDiffEq.jl and their corresponding Butcher tableaux
  if alg == BS3()
    coefficients = (;
      A = [0 0 0 0;
          1/2 0 0 0;
          0 3/4 0 0;
          2/9 1/3 4/9 0],
      b = [2/9, 1/3, 4/9, 0],
      bembd = [7/24, 1/4, 1/3, 1/8],
      c = [0, 1/2, 3/4, 1],
      order_main = 3)
  elseif  alg == Heun()
    coefficients = (;
      A = [0 0 0;
          1 0 0;
          1/2 1/2 0],
      b = [1/2, 1/2, 0],
      bembd = [1.0, 0, 0],
      c = [0.0, 1, 1],
      order_main = 2)
  end

  # initial time step size
  dt, _ = ode_determine_initdt(
    ode.u0, first(ode.tspan), abstol, reltol, ode, coefficients.order_main)

  # Solve the ODE using the plain code below
  sol_residual = @time ode_solve(ode, coefficients;
                                 dt, abstol, reltol,
                                 controller = PIDController(betas...),
                                 estimate, estimate_tolerance,
                                 save_everystep = false)
  @info "residual estimator finished" sol_residual.stats.nf sol_residual.stats.naccept sol_residual.stats.nreject
  GC.gc()

  # Solve the ODE using OrdinaryDiffEq.jl
  saved_values = SavedValues(Float64, typeof(nothing))
  saving_callback = SavingCallback(Returns(nothing), saved_values)
  sol_embedded = @time solve(ode, alg;
                             callback = saving_callback,
                             dt, abstol, reltol,
                             controller = PIDController(betas...),
                             save_everystep = false)
  @info "embedded estimator finished" sol_embedded.stats.nf sol_embedded.stats.naccept sol_embedded.stats.nreject

  errors_residual = analysis_callback(sol_residual)
  errors_embedded = analysis_callback(sol_embedded)
  @info "errors" errors_residual errors_embedded

  if plot_results
    fig_dt = plot(xguide = L"Time $t$", yguide = L"Time step size $\Delta t$")
    plot!(fig_dt, sol_residual.t[begin:(end-1)], diff(sol_residual.t);
          label = "residual", plot_kwargs()...)
    plot!(fig_dt, saved_values.t[begin:(end-1)], diff(saved_values.t);
          label = "embedded", linestyle = :dot, plot_kwargs()...)
  else
    fig_dt = nothing
  end

  return (; fig_dt, errors_residual, errors_embedded)
end

function plot_compressible_euler(; estimate = :residual_cubic_l1)
  estimate_tolerance = 1.0e-6
  figdir = joinpath(dirname(@__DIR__), "figures")
  isdir(figdir) || mkdir(figdir)

  res = test_compressible_euler(; tol = 1.0e-5, estimate, estimate_tolerance)
  savefig(res.fig_dt, joinpath(figdir, "euler_dt_vs_t__BS3_cubicL1_tol1em5.pdf"))

  @info "Results saved in directory figdir" figdir
  return nothing
end



################################################################################
# This struct contains all information required for the simple time stepping
# and mimics the approach of OrdinaryDiffEq at the same time but with
# significantly reduced complexity.
mutable struct Integrator{uType, tType, Prob, RealT, Controller}
  t::tType
  dt::tType
  u::uType
  uembd::uType
  uprev::uType
  utmp::Vector{uType}
  ktmp::Vector{uType}
  prob::Prob
  A::Matrix{RealT}
  b::Vector{RealT}
  bembd::Vector{RealT}
  c::Vector{RealT}
  order_for_control::Int
  abstol::RealT
  reltol::RealT
  controller::Controller
  estimate::Symbol
  estimate_tolerance::RealT
  save_everystep::Bool
  naccept::Int
  nreject::Int
  nf::Int
end

# A simple function applying the explicit Runge-Kutta method defined by
# `coefficients` to tsolve the ODE `prob`.
function ode_solve(prob::ODEProblem, coefficients;
                   dt, abstol, reltol,
                   controller, estimate = :residual_cubic_l1,
                   estimate_tolerance = 1.0e-8,
                   save_everystep = true)
  # initialization
  t = first(prob.tspan)
  (; A, b, bembd, c, order_main) = coefficients
  if estimate === :embedded
    order_for_control = order_main
  else
    order_for_control = order_main + 1
  end
  u = copy(prob.u0)
  uprev = copy(u)
  uembd = similar(u)
  utmp = Vector{typeof(u)}(undef, length(b))
  for i in eachindex(utmp)
    utmp[i] = similar(u)
  end
  ktmp = Vector{typeof(u)}(undef, length(b))
  for i in eachindex(ktmp)
    ktmp[i] = similar(u)
  end
  naccept = 0
  nreject = 0
  nf = 0

  integrator = Integrator(t, dt, u, uembd, uprev, utmp, ktmp, prob,
                          A, b, bembd, c, order_for_control,
                          abstol, reltol, controller,
                          estimate, estimate_tolerance,
                          save_everystep,
                          naccept, nreject, nf)

  sol = (; u = [copy(u)], t = [t], prob)

  # main loop
  solve!(integrator, sol)

  stats = (; naccept = integrator.naccept,
             nreject = integrator.nreject,
             nf = integrator.nf)

  return (; sol..., stats)
end

function solve!(integrator::Integrator, sol)
  (; u, uprev, uembd, utmp, ktmp,
     A, b, bembd, c, order_for_control,
     abstol, reltol, controller,
     estimate, estimate_tolerance, save_everystep) = integrator
  f! = integrator.prob.f.f
  params = integrator.prob.p
  tend = last(integrator.prob.tspan)

  # solve until we have reached the final time
  while integrator.t < tend
    # adjust `dt` at the final time
    if integrator.t + integrator.dt > tend
      integrator.dt = tend - integrator.t
    end

    # compute explicit RK step with step size `dt`
    for i in eachindex(b)
      copy!(utmp[i], uprev)
      for j in 1:i-1
        @. utmp[i] = utmp[i] + (A[i,j] * integrator.dt) * ktmp[j]
      end
      f!(ktmp[i], utmp[i], params, integrator.t + c[i] * integrator.dt)
      integrator.nf += 1
    end
    copy!(u,     uprev)
    copy!(uembd, uprev)
    for i in eachindex(b)
      @. u     = u     + (b[i]     * integrator.dt) * ktmp[i]
      @. uembd = uembd + (bembd[i] * integrator.dt) * ktmp[i]
    end

    # check whether we are finished
    if integrator.t + integrator.dt ≈ tend
      integrator.t += integrator.dt
      integrator.naccept += 1
      copy!(uprev, u)
      push!(sol.u, copy(u))
      push!(sol.t, integrator.t)
      break
    end

    # compute error estimate
    if estimate === :embedded
      error_estimate = compute_error_estimate(u, uprev, uembd, abstol, reltol)
    elseif estimate === :residual_quadratic_l1
      error_estimate = let u0 = utmp[begin], du0 = ktmp[begin], u1 = utmp[end], du1 = ktmp[end], dt = integrator.dt
        # Left-biased quadratic Hermite interpolation
        function uhat!(u, t)
          @. u = (1 - t^2 / dt^2) * u0 +
                 (t - t^2 / dt) * du0 +
                 (t^2 / dt^2) * u1
          return nothing
        end

        function duhat!(du, t)
          @. du = (-2 * t / dt^2) * u0 +
                  (1 - 2 * t / dt) * du0 +
                  (2 * t / dt^2) * u1
          return nothing
        end

        du = zero(u)
        u_interpolated = zero(u)
        du_interpolated = zero(u)
        integral, err = quadgk(0.0, dt; rtol = estimate_tolerance, norm = @fastmath(norm)) do t
          uhat!(u_interpolated, t)
          f!(du, u_interpolated, params, integrator.t + t)
          duhat!(du_interpolated, t)
          du_interpolated .-= du
          return @fastmath norm(du_interpolated)
        end

        integral / (abstol + reltol * max(norm(u0), norm(u1)))
      end
    elseif estimate === :residual_cubic_l1
      error_estimate = let u0 = utmp[begin], du0 = ktmp[begin], u1 = utmp[end], du1 = ktmp[end], dt = integrator.dt
        # Central cubic Hermite interpolation
        function uhat!(u, t)
          @. u = (1 - 3 * t^2 / dt^2 + 2 * t^3 / dt^3) * u0 +
                (t - 2 * t^2 / dt + t^3 / dt^2) * du0 +
                (3 * t^2 / dt^2 - 2 * t^3 / dt^3) * u1 +
                (-t^2 / dt + t^3 / dt^2) * du1
          return nothing
        end

        function duhat!(du, t)
          @. du = (-6 * t / dt^2 + 6 * t^2 / dt^3) * u0 +
                  (1 - 4 * t / dt + 3 * t^2 / dt^2) * du0 +
                  (6 * t / dt^2 - 6 * t^2 / dt^3) * u1 +
                  (-2 * t / dt + 3 * t^2 / dt^2) * du1
          return nothing
        end

        du = zero(u)
        u_interpolated = zero(u)
        du_interpolated = zero(u)
        integral, err = quadgk(0.0, dt; rtol = estimate_tolerance, norm = @fastmath(norm)) do t
          uhat!(u_interpolated, t)
          f!(du, u_interpolated, params, integrator.t + t)
          duhat!(du_interpolated, t)
          du_interpolated .-= du
          return @fastmath norm(du_interpolated)
        end

        integral / (abstol + reltol * max(norm(u0), norm(u1)))
      end
    elseif estimate === :residual_quadratic_l2
      error_estimate = let u0 = utmp[begin], du0 = ktmp[begin], u1 = utmp[end], du1 = ktmp[end], dt = integrator.dt
        # Left-biased quadratic Hermite interpolation
        function uhat!(u, t)
          @. u = (1 - t^2 / dt^2) * u0 +
                 (t - t^2 / dt) * du0 +
                 (t^2 / dt^2) * u1
          return nothing
        end

        function duhat!(du, t)
          @. du = (-2 * t / dt^2) * u0 +
                  (1 - 2 * t / dt) * du0 +
                  (2 * t / dt^2) * u1
          return nothing
        end

        du = zero(u)
        u_interpolated = zero(u)
        du_interpolated = zero(u)
        integral, err = quadgk(0.0, dt; rtol = estimate_tolerance, norm = @fastmath(norm)) do t
          uhat!(u_interpolated, t)
          f!(du, u_interpolated, params, integrator.t + t)
          duhat!(du_interpolated, t)
          du_interpolated .-= du
          return @fastmath norm(du_interpolated)^2
        end

        sqrt(dt * integral) / (abstol + reltol * max(norm(u0), norm(u1)))
      end
    elseif estimate === :residual_cubic_l2
      error_estimate = let u0 = utmp[begin], du0 = ktmp[begin], u1 = utmp[end], du1 = ktmp[end], dt = integrator.dt
        # Central cubic Hermite interpolation
        function uhat!(u, t)
          @. u = (1 - 3 * t^2 / dt^2 + 2 * t^3 / dt^3) * u0 +
                (t - 2 * t^2 / dt + t^3 / dt^2) * du0 +
                (3 * t^2 / dt^2 - 2 * t^3 / dt^3) * u1 +
                (-t^2 / dt + t^3 / dt^2) * du1
          return nothing
        end

        function duhat!(du, t)
          @. du = (-6 * t / dt^2 + 6 * t^2 / dt^3) * u0 +
                  (1 - 4 * t / dt + 3 * t^2 / dt^2) * du0 +
                  (6 * t / dt^2 - 6 * t^2 / dt^3) * u1 +
                  (-2 * t / dt + 3 * t^2 / dt^2) * du1
          return nothing
        end

        du = zero(u)
        u_interpolated = zero(u)
        du_interpolated = zero(u)
        integral, err = quadgk(0.0, dt; rtol = estimate_tolerance, norm = @fastmath(norm)) do t
          uhat!(u_interpolated, t)
          f!(du, u_interpolated, params, integrator.t + t)
          duhat!(du_interpolated, t)
          du_interpolated .-= du
          return @fastmath norm(du_interpolated)^2
        end

        sqrt(dt * integral) / (abstol + reltol * max(norm(u0), norm(u1)))
      end
    end

    # adapt `dt`
    dt_factor = compute_dt_factor!(controller, error_estimate, order_for_control)

    # accept or reject the step and update `dt`
    if accept_step(controller, dt_factor)
      accept_step!(controller)
      integrator.naccept += 1
      integrator.t += integrator.dt
      copy!(uprev, u)
      if save_everystep
        push!(sol.u, copy(u))
      end
      push!(sol.t, integrator.t)
    else
      reject_step!(controller)
      integrator.nreject += 1
    end
    integrator.dt *= dt_factor

    if integrator.dt < 1.0e-14
      @error "time step too small" integrator.dt integrator.t integrator.naccept integrator.nreject
      error()
    end
  end

  return integrator
end



function compute_error_estimate(u, uprev, uembd, abstol, reltol)
  err = zero(real(eltype(u)))
  err_n = 0
  for i in eachindex(u)
    tol = abstol + reltol * max(abs(u[i]), abs(uprev[i]))
    if tol > 0
      err += abs2(u[i] - uembd[i]) / tol^2
      err_n += 1
    end
  end
  return sqrt(err / err_n)
end


#=
A PID controller is of the form

struct PIDController{QT, Limiter} <: AbstractController
  beta::Vector{QT}  # controller coefficients (length 3)
  err ::Vector{QT}  # history of the error estimates (length 3)
  accept_safety::QT # accept a step if the predicted change of the step size
                    # is bigger than this parameter
  limiter::Limiter  # limiter of the dt factor (before clipping)
end

with

default_dt_factor_limiter(x) = one(x) + atan(x - one(x))

as default `limiter`, `accept_safety=0.81` as default value. The vector `beta`
contains the coefficients βᵢ of the PID controller. The vector `err` stores
error estimates.
=#
function compute_dt_factor!(controller::PIDController, error_estimate, order_for_control)
  beta1, beta2, beta3 = controller.beta
  controller.err[1] = inv(error_estimate)
  err1, err2, err3 = controller.err

  # If the error estimate is zero, we can increase the step size as much as
  # desired. This additional check fixes problems of the code below when the
  # error estimates become zero
  # -> err1, err2, err3 become Inf
  # -> err1^positive_number * err2^negative_number becomes NaN
  # -> dt becomes NaN
  #
  # `error_estimate_min` is smaller than PETSC_SMALL used in the equivalent logic in PETSc.
  # For example, `eps(Float64) ≈ 2.2e-16` but `PETSC_SMALL ≈ 1.0e-10` for `double`.
  error_estimate_min = eps(real(typeof(error_estimate)))
  # The code below is a bit more robust than
  # ```
  # if iszero(error_estimate)
  #   error_estimate = eps(typeof(error_estimate))
  # end
  # ```
  error_estimate = ifelse(error_estimate > error_estimate_min,
                          error_estimate, error_estimate_min)

  k = order_for_control
  dt_factor = err1^(beta1 / k) * err2^(beta2 / k) * err3^(beta3 / k)

  if isnan(dt_factor)
    @warn "unlimited dt_factor" dt_factor err1 err2 err3 beta1 beta2 beta3 k controller.err[1] controller.err[2] controller.err[3]
    error()
  end
  dt_factor = controller.limiter(dt_factor)

  return dt_factor
end

function accept_step(controller::PIDController, dt_factor)
  return dt_factor >= controller.accept_safety
end

function accept_step!(controller::PIDController)
  controller.err[3] = controller.err[2]
  controller.err[2] = controller.err[1]
  return nothing
end

function reject_step!(controller::PIDController)
  return nothing
end

function ode_determine_initdt(
      u0, t, abstol, reltol,
      prob::DiffEqBase.AbstractODEProblem{uType, tType},
      order) where {tType, uType}
  tdir = true
  _tType = eltype(tType)
  f = prob.f
  p = prob.p
  oneunit_tType = oneunit(_tType)
  dtmax = last(prob.tspan) - first(prob.tspan)
  dtmax_tdir = tdir * dtmax

  dtmin = nextfloat(OrdinaryDiffEq.DiffEqBase.prob2dtmin(prob))
  smalldt = convert(_tType, oneunit_tType * 1 // 10^(6))

  sk = @. abstol + abs(u0) * reltol


  f₀ = similar(u0)
  f(f₀, u0, p, t)

  tmp = @. u0 / sk
  d₀ = norm(tmp)

  @. tmp = f₀ / sk * oneunit_tType
  d₁ = norm(tmp)

  if d₀ < 1//10^(5) || d₁ < 1//10^(5)
    dt₀ = smalldt
  else
    dt₀ = convert(_tType,oneunit_tType*(d₀/d₁)/100)
  end
  dt₀ = min(dt₀, dtmax_tdir)

  if typeof(one(_tType)) <: AbstractFloat && dt₀ < 10eps(_tType) * oneunit(_tType)
    return tdir * smalldt
  end

  dt₀_tdir = tdir * dt₀

  u₁ = zero(u0)
  @. u₁ = u0 + dt₀_tdir * f₀
  f₁ = zero(f₀)
  f(f₁, u₁, p, t + dt₀_tdir)

  f₀ == f₁ && return tdir * max(dtmin, 100 * dt₀)

  @. tmp = (f₁ - f₀) / sk * oneunit_tType
  d₂ = norm(tmp) / dt₀ * oneunit_tType

  max_d₁d₂ = max(d₁, d₂)
  if max_d₁d₂ <= 1 // Int64(10)^(15)
    dt₁ = max(convert(_tType, oneunit_tType * 1 // 10^(6)), dt₀ * 1 // 10^(3))
  else
    dt₁ = convert(_tType,
                  oneunit_tType *
                  10.0^(-(2 + log10(max_d₁d₂)) / order))
  end
  return tdir * max(dtmin, min(100 * dt₀, dt₁, dtmax_tdir)), f₀
end
