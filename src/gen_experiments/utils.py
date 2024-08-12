import logging
from itertools import chain
from typing import Annotated, TypedDict, cast
from warnings import warn

import auto_ks as aks
import kalman
import numpy as np
import pysindy as ps
import sklearn
import sklearn.metrics
from numpy.typing import NDArray

from .typing import Float1D, Float2D, FloatND

logger = logging.getLogger(__name__)


class SINDyTrialData(TypedDict):
    dt: float
    coeff_true: Annotated[Float2D, "(n_coord, n_features)"]
    coeff_fit: Annotated[Float2D, "(n_coord, n_features)"]
    feature_names: Annotated[list[str], "length=n_features"]
    input_features: Annotated[list[str], "length=n_coord"]
    t_train: Float1D
    x_train: np.ndarray
    x_true: np.ndarray
    smooth_train: np.ndarray
    x_test: np.ndarray
    x_dot_test: np.ndarray
    model: ps.SINDy


class SINDyTrialUpdate(TypedDict):
    t_sim: Float1D
    t_test: Float1D
    x_sim: FloatND


class FullSINDyTrialData(SINDyTrialData):
    t_sim: Float1D
    x_sim: np.ndarray


def diff_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "finitedifference":
        return ps.FiniteDifference
    if normalized_kind == "smoothedfinitedifference":
        return ps.SmoothedFiniteDifference
    elif normalized_kind == "sindy":
        return ps.SINDyDerivative
    else:
        raise ValueError


def feature_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind is None:
        return ps.PolynomialLibrary
    elif normalized_kind == "polynomial":
        return ps.PolynomialLibrary
    elif normalized_kind == "fourier":
        return ps.FourierLibrary
    elif normalized_kind == "weak":
        return ps.WeakPDELibrary
    elif normalized_kind == "pde":
        return ps.PDELibrary
    else:
        raise ValueError


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    elif normalized_kind == "miosr":
        return ps.MIOSR
    elif normalized_kind == "trap":
        return ps.TrappingSR3
    elif normalized_kind == "ensemble":
        return ps.EnsembleOptimizer
    else:
        raise ValueError


def coeff_metrics(coefficients, coeff_true):
    metrics = {}
    metrics["coeff_precision"] = sklearn.metrics.precision_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_recall"] = sklearn.metrics.recall_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_f1"] = sklearn.metrics.f1_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_mse"] = sklearn.metrics.mean_squared_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["coeff_mae"] = sklearn.metrics.mean_absolute_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["main"] = metrics["coeff_f1"]
    return metrics


def integration_metrics(model, x_test, t_train, x_dot_test):
    metrics = {}
    metrics["mse-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_squared_error,
    )
    metrics["mae-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_absolute_error,
    )
    return metrics


def unionize_coeff_matrices(
    model: ps.SINDy, coeff_true: list[dict[str, float]]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Reformat true coefficients and coefficient matrix compatibly

    In order to calculate accuracy metrics between true and estimated
    coefficients, this function compares the names of true coefficients
    and a the fitted model's features in order to create comparable
    (i.e. non-ragged) true and estimated coefficient matrices.  In
    a word, it stacks the correct coefficient matrix and the estimated
    coefficient matrix in a matrix that represents the union of true
    features and modeled features.

    Arguments:
        model: fitted model
        coeff_true: list of dicts of format function_name: coefficient,
            one dict for each modeled coordinate/target

    Returns:
        Tuple of true coefficient matrix, estimated coefficient matrix,
        and combined feature names

    Warning:
        Does not disambiguate between commutatively equivalent function
        names such as 'x z' and 'z x' or 'x^2' and 'x x'
    """
    model_features = model.get_feature_names()
    true_features = [set(coeffs.keys()) for coeffs in coeff_true]
    unmodeled_features = set(chain.from_iterable(true_features)) - set(model_features)
    model_features.extend(list(unmodeled_features))
    est_coeff_mat = model.coefficients()
    new_est_coeff = np.zeros((est_coeff_mat.shape[0], len(model_features)))
    new_est_coeff[:, : est_coeff_mat.shape[1]] = est_coeff_mat
    true_coeff_mat = np.zeros_like(new_est_coeff)
    for row, terms in enumerate(coeff_true):
        for term, coeff in terms.items():
            true_coeff_mat[row, model_features.index(term)] = coeff

    return true_coeff_mat, new_est_coeff, model_features


def make_model(
    input_features: list[str],
    dt: float,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
) -> ps.SINDy:
    """Build a model with object parameters dictionaries

    e.g. {"kind": "finitedifference"} instead of FiniteDifference()
    """

    def finalize_param(lookup_func, pdict, lookup_key):
        try:
            cls_name = pdict.pop(lookup_key)
        except AttributeError:
            cls_name = pdict.vals.pop(lookup_key)
            pdict = pdict.vals

        param_cls = lookup_func(cls_name)
        param_final = param_cls(**pdict)
        pdict[lookup_key] = cls_name
        return param_final

    diff = finalize_param(diff_lookup, diff_params, "diffcls")
    features = finalize_param(feature_lookup, feat_params, "featcls")
    opt = finalize_param(opt_lookup, opt_params, "optcls")
    return ps.SINDy(
        differentiation_method=diff,
        optimizer=opt,
        t_default=dt,  # type: ignore
        feature_library=features,
        feature_names=input_features,
    )


def simulate_test_data(model: ps.SINDy, dt: float, x_test: Float2D) -> SINDyTrialUpdate:
    """Add simulation data to grid_data

    This includes the t_sim and x_sim keys.  Does not mutate argument.
    Returns:
        Complete GridPointData
    """
    t_test = cast(Float1D, np.arange(0, len(x_test) * dt, step=dt))
    t_sim = t_test
    try:
        x_sim = cast(Float2D, model.simulate(x_test[0], t_test))
    except ValueError:
        warn(message="Simulation blew up; returning zeros")
        x_sim = np.zeros_like(x_test)
    # truncate if integration returns wrong number of points
    t_sim = cast(Float1D, t_test[: len(x_sim)])
    return {"t_sim": t_sim, "x_sim": x_sim, "t_test": t_test}


def kalman_generalized_cv(
    times: np.ndarray, measurements: np.ndarray, alpha0: float = 1, alpha_max=1e12
):
    """Find kalman parameter alpha using GCV error

    See Boyd & Barratt, Fitting a Kalman Smoother to Data.  No regularization
    """
    measurements = measurements.reshape((-1, 1))
    dt = times[1] - times[0]
    Ai = np.array([[1, 0], [dt, 1]])
    Qi = kalman.gen_Qi(dt)
    Qi_rt_inv = np.linalg.cholesky(np.linalg.inv(Qi))
    Qi_r_i_vec = np.reshape(Qi_rt_inv, (-1, 1))
    Qi_proj = (
        lambda vec: Qi_r_i_vec
        @ (Qi_r_i_vec.T @ Qi_r_i_vec) ** -1
        @ (Qi_r_i_vec.T)
        @ vec
    )
    Hi = np.array([[0, 1]])
    Ri = np.eye(1)
    Ri_rt_inv = Ri
    params0 = aks.KalmanSmootherParameters(Ai, Qi_rt_inv, Hi, Ri)
    mask = np.ones_like(measurements, dtype=bool)
    mask[::4] = False

    def proj(curr_params, t):
        W_n_s_v = np.reshape(curr_params.W_neg_sqrt, (-1, 1))
        W_n_s_v = np.reshape(Qi_proj(W_n_s_v), (2, 2))
        new_params = aks.KalmanSmootherParameters(Ai, W_n_s_v, Hi, Ri_rt_inv)
        return new_params, t

    params, info = aks.tune(params0, proj, measurements, K=mask, lam=0.1, verbose=False)
    est_Q = np.linalg.inv(params.W_neg_sqrt @ params.W_neg_sqrt.T)
    est_alpha = 1 / (est_Q / Qi).mean()
    if est_alpha < 1 / alpha_max:
        logger.warn(
            f"Kalman GCV estimated alpha escaped bounds, assigning {1 / alpha_max}"
        )
        return 1 / alpha_max
    elif est_alpha > alpha_max:
        logger.warn(f"Kalman GCV estimated alpha escaped bounds, assigning {alpha_max}")
        return alpha_max
    elif np.isnan(est_alpha):
        raise ValueError("GCV Failed")
    logger.info(f"Identified best alpha for Kalman GCV: {est_alpha}")
    return est_alpha
