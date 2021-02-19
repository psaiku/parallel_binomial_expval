include("opt_berpaths_mc_serial.jl")

n_sim =131072
tic()
optval = cal_optvalue(32, 5, 0.3, 10, 0.06, 1, 1, n_sim)
@printf("MC Est. : %f, Runtime: %s\n", optval, toc())
