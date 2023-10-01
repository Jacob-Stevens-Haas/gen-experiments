import numpy as np
from pysindy.differentiation import SpectralDerivative
from .utils import (
    gen_pde_data,
    compare_coefficient_plots,
    plot_pde_training_data,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model,
)

def diffuse1D(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 0
    u[-1] = 0
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    return np.reshape(uxx, nx)

def diffuse2D(t, u, dx, nx):
    u = np.reshape(u, (nx, nx))
    u[0,:] = 100
    u[-1, :] = 1000
    u[:, -1] = 500
    u[:, 0] = 300
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uyy = SpectralDerivative(d=2, axis=1)._differentiate(u, dx)
    return np.reshape(uxx+uyy, (nx**2,))

def diffuse3D(t, u, dx, nx):
    u = np.reshape(u, (nx, nx, nx))
    u[0,:,:] = 1000
    u[-1,:,:] = 2000
    u[:,0,:] = 1500
    u[:,-1,:] = 2500
    u[:,:,0] = 1250
    u[:,:,-1] = 2250
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uyy = SpectralDerivative(d=2, axis=1)._differentiate(u, dx)
    uzz = SpectralDerivative(d=2, axis=2)._differentiate(u, dx)
    return np.reshape(uxx+uyy+uzz, (nx**3,))

def burgers1D(t, u, dx, nx):
    u = np.reshape(u, nx)
    u[0] = 1000
    u[-1] = 200
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    return np.reshape((uxx - u*ux), nx)

def burgers2D(t, u, dx, nx):
    u = np.reshape(u, (nx, nx))
    u[0,:] = 100
    u[-1, :] = 1000
    u[:, -1] = 500
    u[:, 0] = 300
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uyy = SpectralDerivative(d=2, axis=1)._differentiate(u, dx)
    uy = SpectralDerivative(d=1, axis=1)._differentiate(u, dx)
    return  np.reshape((uxx + uyy - u*ux - u*uy), (nx**2,))

def burgers3D(t, u, dx, nx):
    u = np.reshape(u, (nx, nx, nx))
    u[0,:,:] = 1000
    u[-1,:,:] = 2000
    u[:,0,:] = 1500
    u[:,-1,:] = 2500
    u[:,:,0] = 1250
    u[:,:,-1] = 2250
    sf = 1e-5
    uxx = sf * SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    ux = sf * SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uyy = sf * SpectralDerivative(d=2, axis=1)._differentiate(u, dx)
    uy = sf * SpectralDerivative(d=1, axis=1)._differentiate(u, dx)
    uzz = sf * SpectralDerivative(d=2, axis=2)._differentiate(u, dx)
    uz = sf * SpectralDerivative(d=1, axis=2)._differentiate(u, dx)    
    return  np.reshape((uxx + uyy + uzz - u*ux - u*uy - u*uz), (nx**3,))/sf

def ks(t, u ,dx, nx):
    u = np.reshape(u, nx)
    u[0] = 100
    u[-1] = 200
    ux = SpectralDerivative(d=1, axis=0)._differentiate(u, dx)
    uxx = SpectralDerivative(d=2, axis=0)._differentiate(u, dx)
    uxxxx = SpectralDerivative(d=4, axis=0)._differentiate(u, dx)
    return np.reshape(-uxx-uxxxx-u*ux, nx)

pde_setup = {
    "diffuse1D": {
        "rhsfunc": {
            "func": diffuse1D,
            "dimension": 1
        },
        "input_features": ["u"],
        "initial_condition": 10*np.exp(-(np.arange(0, 10, 0.1)-5)**2/2),
        "spatial_args": [0.1, 100],
        "time_args": [0.1, 10],
        "coeff_true": [
            {"u_11": 1}
        ],
        "spatial_grid": np.arange(0, 10, 0.1)
    },
    "diffuse2D": {
        "rhsfunc": {
            "func": diffuse2D,
            "dimension": 2
        },
        "input_features": ["uxx", "uyy"],
        "initial_condition": np.reshape(np.zeros((10, 10)), (100,)),
        "spatial_args": [0.01, 100],
        "time_args": [0.01, 1],
        "coeff_true": [
            {"u_11": 1}
        ],
        "spatial_grid": np.arange(0, 10, 0.1)
    },
    "diffuse3D": {
        "rhsfunc": {
            "func": diffuse3D,
            "dimension": 3
        },
        "input_features": ["u", "uxx", "uyy", "uzz"],
        "initial_condition": np.reshape(np.zeros((10, 10, 10)), (1000,)),
        "spatial_args": [0.01, 10],
        "time_args": [0.01, 10]
    },
    "burgers1D": {
        "rhsfunc": {
            "func": burgers1D,
            "dimension": 1
        },
        "input_features": ["u", "ux", "uxx"],
        "initial_condition": np.exp(-(np.linspace(0, 1, 100)-3)/2),
        "spatial_args": [0.01, 100],
        "time_args": [0.01, 100]
    },
    "burgers2D": {
        "rhsfunc": {
            "func": burgers2D,
            "dimension": 2
        },
        "input_features": ["u", "ux", "uxx", "uy", "uyy"],
        "initial_condition": np.reshape(np.zeros((10, 10)), (100,)),
        "spatial_args": [0.01, 10],
        "time_args": [0.01, 10]
    },
    "burgers3D": {
        "rhsfunc": {
            "func": burgers3D,
            "dimension": 3
        },
        "input_features": ["u", "ux", "uxx", "uy", "uyy", "uz", "uzz"],
        "initial_condition": np.reshape(np.zeros((10, 10, 10)), (1000,)),
        "spatial_args": [0.01, 10],
        "time_args": [0.01, 10]
    },
    "kuramoto-sivashinsky": {
        "rhsfunc": {
            "func": ks,
            "dimension": 1
        },
        "input_features": ["u", "ux", "uxx", "uxxxx"],
        "initial_condition": np.zeros(100),
        "spatial_args": [0.01, 100],
        "time_args": [0.01, 100]
    }
}

def run_pde(
    seed: float,
    /,
    group: str,
    # sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = True,
) -> dict:
    rhsfunc = pde_setup[group]["rhsfunc"]["func"]
    input_features = pde_setup[group]["input_features"]
    initial_condition = pde_setup[group]["initial_condition"]
    spatial_args = pde_setup[group]["spatial_args"]
    time_args = pde_setup[group]["time_args"]
    dimension = pde_setup[group]["rhsfunc"]["dimension"]
    coeff_true = pde_setup[group]["coeff_true"]
    spatial_grid = pde_setup[group]["spatial_grid"]
    try:
        time_args = pde_setup[group]["time_args"]
    except KeyError:
        time_args = [0.01, 10]
    nonnegative = False
    dt, t_train, x_train, x_test, x_dot_test, x_train_true = gen_pde_data(
        rhsfunc,
        initial_condition,
        spatial_args,
        dimension,
        seed,
        noise_abs=0,
        dt=time_args[0],
        t_end=time_args[1]
    )
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train[0], quiet=True)
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    # make the plots
    if display:
        model.print()
        compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )
        smoothed_last_train = model.differentiation_method.smoothed_x_
        plot_pde_training_data(x_train[-1], x_train_true, smoothed_last_train)

    # calculate metrics
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return (
            metrics, {
                "t_train": t_train,
                "x_train": x_train,
                "x_test": x_test,
                "x_dot_test": x_dot_test,
                "x_train_true": x_train_true,
                "model": model,
            }
        )
    return metrics