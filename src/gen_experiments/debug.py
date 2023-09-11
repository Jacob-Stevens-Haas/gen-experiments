import gen_experiments
import numpy as np
import pysindy as ps
from pysindy.differentiation import SpectralDerivative
from gen_experiments.utils import (
    gen_data, _make_model, unionize_coeff_matrices, 
    compare_coefficient_plots, plot_pde_training_data, coeff_metrics, 
    integration_metrics, plot_pde_test_trajectories
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
rhsfunc = pde_setup["diffuse1D"]["rhsfunc"]["func"]
input_features = pde_setup["diffuse1D"]["input_features"]
initial_condition = pde_setup["diffuse1D"]["initial_condition"]
spatial_args = pde_setup["diffuse1D"]["spatial_args"]
time_args = pde_setup["diffuse1D"]["time_args"]
dimension = pde_setup["diffuse1D"]["rhsfunc"]["dimension"]
coeff_true = pde_setup["diffuse1D"]["coeff_true"]
spatial_grid = pde_setup["diffuse1D"]["spatial_grid"]
nonnegative = False
dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_pde_data(
    rhsfunc,
    initial_condition,
    spatial_args,
    dimension,
    seed=42,
    nonnegative=nonnegative,
    dt=time_args[0],
    t_end=time_args[1]
)

print(x_train[0].shape, x_test[0].shape, x_dot_test[0].shape, x_train_true.shape)
# x_train = x_train[0].flatten()
# ########## PDE Library ##########
library_functions = [lambda x: x]
library_function_names = [lambda x: x]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=2,
    spatial_grid=spatial_grid,
)

opt = ps.STLSQ()
model = ps.SINDy(feature_library=pde_lib, optimizer=opt, feature_names=["u"])
model.fit(x_train[0], t=dt)
model.print()
coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)
compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )
smoothed_last_train = model.differentiation_method.smoothed_x_
plot_pde_training_data(x_train[-1], x_train_true[-1], smoothed_last_train)
plot_pde_test_trajectories(x_test[-1], model, dt)
metrics = coeff_metrics(coefficients, coeff_true)
metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
print(metrics)
pass
######## Model ########
# diff_params = gen_experiments.diff_params["test2"]
# feat_params = gen_experiments.feat_params["test2"]
# opt_params = gen_experiments.opt_params["test"]
# model = _make_model(input_features, dt, diff_params, feat_params, opt_params)
# print(model)
# model.fit(x_train[0])
# model.print()
# model.coefficients()
# model.predict(x_test[0])
# print(x_dot_test)
######## 1D Plot ########
# figure1 = plt.figure()
# plt.imshow(x_train[0])
# # plt.colorbar()
# plt.show()

######### 3D Plot #########
# Nx, Ny, Nz, Nt, Nu = x_train[0].shape
# x_coords = np.linspace(0, Nx - 1, Nx)
# y_coords = np.linspace(0, Ny - 1, Ny)
# z_coords = np.linspace(0, Nz - 1, Nz)
# t_coords = np.linspace(0, Nt - 1, Nt)
# u_coords = np.linspace(0, Nu - 1, Nu)

# x_coords, y_coords, z_coords, t_coords, u_coords = np.meshgrid(x_coords, y_coords, z_coords, t_coords, u_coords, indexing='ij')

# # Flatten the coordinates and the data for scatter plot
# x_flat = x_coords.flatten()
# y_flat = y_coords.flatten()
# z_flat = z_coords.flatten()
# t_flat = t_coords.flatten()
# u_flat = u_coords.flatten()
# data_flat = x_train[0].flatten()

# # Create a scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(x_flat, y_flat, z_flat, c=data_flat, cmap='viridis')
# fig.colorbar(sc)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Visualization of 5D Data')
# plt.show()

########## Run Function ##########
# run_pde(42, group="diffuse1D",
#         diff_params=gen_experiments.diff_params["test"],
#         feat_params=gen_experiments.feat_params["test"],
#         opt_params=gen_experiments.opt_params["test"]
#         )

######### PYSINDY ODEs #########
# lorenz = lambda z,t : [10*(z[1] - z[0]),
#                        z[0]*(28 - z[2]) - z[1],
#                        z[0]*z[1] - 8/3*z[2]]
# t = np.arange(0,2,.002)
# x= integrate.solve_ivp(lorenz, [-8,8,27], t)
# model = ps.SINDy()
# model.fit(x, t=t[1]-t[0])
# model.print()
# model.coefficients()