# %%
import gen_experiments
import numpy as np
import pysindy as ps
from pysindy.differentiation import SpectralDerivative
from gen_experiments.utils import (
    gen_data, _make_model, unionize_coeff_matrices, 
    compare_coefficient_plots, plot_pde_training_data, coeff_metrics, 
    integration_metrics
)
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from gen_experiments.pdes import gen_pde_data, run_pde, pde_setup
from matplotlib.animation import FuncAnimation
from scipy import integrate

######### ODES ##########
# gen_experiments.odes.run(
#     34,
#     group="vdp",
#     sim_params=gen_experiments.sim_params["test"],
#     diff_params=gen_experiments.diff_params["test"],
#     feat_params=gen_experiments.feat_params["test"],
#     opt_params=gen_experiments.opt_params["test"],
# )

######### PDES #########
# rhsfunc = pde_setup["diffuse1D"]["rhsfunc"]["func"]
# input_features = pde_setup["diffuse1D"]["input_features"]
# initial_condition = pde_setup["diffuse1D"]["initial_condition"]
# spatial_args = pde_setup["diffuse1D"]["spatial_args"]
# time_args = pde_setup["diffuse1D"]["time_args"]
# dimension = pde_setup["diffuse1D"]["rhsfunc"]["dimension"]
# coeff_true = pde_setup["diffuse1D"]["coeff_true"]
# spatial_grid = pde_setup["diffuse1D"]["spatial_grid"]
# nonnegative = False
# dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_pde_data(
#     rhsfunc,
#     initial_condition,
#     spatial_args,
#     dimension,
#     seed=42,
#     noise_abs=0,
#     dt=time_args[0],
#     t_end=time_args[1]
# )

# print(x_train[0].shape, x_test[0].shape, x_dot_test[0].shape, x_train_true.shape)
# ########### PDE Library ##########
# library_functions = [lambda x: x]
# library_function_names = [lambda x: x]
# pde_lib = ps.PDELibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
#     derivative_order=2,
#     spatial_grid=spatial_grid, 
# )
# opt = ps.STLSQ()
# # diff = ps.SINDyDerivative(kind="kalman", alpha=0.05)
# model = ps.SINDy(differentiation_method=None, feature_library=pde_lib, optimizer=opt, feature_names=["u"])
# model.fit(x_train[0], t=dt)
# model.print() 
# coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)
# compare_coefficient_plots(
#             coefficients,
#             coeff_true,
#             input_features=input_features,
#             feature_names=feature_names,
#         )
# smoothed_last_train = model.differentiation_method.smoothed_x_
# plot_pde_training_data(x_train[-1], x_train_true, smoothed_last_train)
# metrics = coeff_metrics(coefficients, coeff_true)
# metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
# print(metrics)
# pass                                          


# %%

run_pde(42, group="diffuse1D", 
        diff_params=gen_experiments.diff_params["test_axis"], 
        opt_params=gen_experiments.opt_params["test"], 
        feat_params=gen_experiments.feat_params["pde"])
# %%
