from sasmodels.core import load_model
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel
from sklearn import metrics
import numpy as np
from saxs_data_processing import io
from saxs_data_processing.sasview_fitting import fit_power_law

# from apdist import AmplitudePhaseDistance as dist
from apdist.distances import AmplitudePhaseDistance as dist
import torch


def target_intensities(q, r, pdi, sld_particle, sld_solvent):
    """
    Calculate target scattering intensity for a given q array, radius (angstroms), and pdi
    """
    kernel = load_model("sphere")
    data = empty_data1D(q)
    model = DirectModel(data, kernel)

    I = model(
        radius=r,
        radius_pd=pdi,
        sld=sld_particle,
        sld_solvent=sld_solvent,
        radius_pd_type="gaussian",
        radius_pd_n=35,
        radius_pd_nsigma=3,
    )

    return I


def rmse_distance(measured_I, target_I, log=True):
    """
    Calculate the RMSE distance between target and measured curves.
    IF log=True, (default), calculates RMSE between log10 of measured and target

    :param measured_I: measured intensity
    :type measured_I: np array
    :param target_I: target intensity to calculate distance to
    :type target_I: np array
    :param log: take log of measured I and target I?
    :type log: bool
    """
    if log:
        measured_I = np.log10(measured_I)
        target_I = np.log10(target_I)

    return metrics.root_mean_squared_error(target_I, measured_I)


def calculate_distance_powerlawscreen(
    data_powerlaw,
    data_rmse,
    target_I,
    cutoff_chisq=200,
    power_law_value=5,
    normalize=True,
):
    """
    Return RMSE distance for data fit poorly by power law fit, constant value for those fit well by power law.

    :param data: brenden df data of measured scattering
    :type data: pd dataframe with q/I/sig cols
    :param target_I: target intensity on same q as data
    :type target_I: np array
    :param cutoff_chisq: cutoff threshold for a 'good' power law fit based on chi-square metric. Data with fits less than this wil be assigned distance 'power_law_value'
    :type cutoff_chisq: int, default 200
    :param power_law_value: value to assign 'good' power law fits instead of an RMSE distance
    :type power_law_value: int, default 5
    :param normalize: if true, normalize first I value of target and data to 1. Default true
    :type normalize: bool
    """

    data_sas = io.df_to_sasdata(data_powerlaw)
    # print(data_sas)
    power_results = fit_power_law(data_sas)

    if float(power_results[0]["chisq"].split("(")[0]) < cutoff_chisq:
        distance = power_law_value
    else:
        meas_I = data_rmse["I"]
        if normalize:
            target_I = target_I / target_I[0]
            meas_I = meas_I / meas_I.iloc[0]

        # drop negative values
        drop_ind = meas_I[meas_I < 0].index
        meas_I = meas_I.drop(drop_ind)
        target_I = np.delete(target_I, drop_ind)

        distance = rmse_distance(meas_I, target_I, log=True)

    return distance


def ap_distance(q_grid, I_measured, I_target, optim="DP", grid_dim=10):
    """
    Calculate amplitude-phase distance between I_measured and I_target.

    Read more on amplitude-phase distance here: https://github.com/kiranvad/Amplitude-Phase-Distance

    I_measured and I_target should share q_grid q vector. q_grid and intensities should be in log space. I_measured should be denoised and smooothed. I_measured and I_target should be scaled onto each other.

    :param q_grid: Linear, evenly spaced q-vector grid in log10(q) space. ie, np.linspace(np.log10(q_min), np.log10(q_max)) Do not use q from measurement/instrument - q
    :type q_grid: np array
    :param I_measured: Measured intensity. Evaluated on q_grid (ie through a Bspline interpolation) and in log10(I) space.
    :type I_measured: np array
    :param I_target: Calculated target scattering intensity
    :type I_target: np array
    :param optim: apdist optimizer type for SRSF minimization. "DP" or "RLBFGS"
    :type optim: str
    :param grid_dim: ap dist grid_dim param. Default 10
    :type grid_dim: int
    :return amplitude: Amplitude (y-axis variation) distance
    :rtype amplitude: float
    :return phase: Phase (x-axis variation) distance
    :rtype phase: float
    """

    assert (
        len(set((len(q_grid), len(I_measured), len(I_target)))) == 1
    ), "q_grid, I_measured, and I_target all need to be the same length"

    amplitude, phase = dist(
        q_grid,
        I_measured,
        I_target,
        optim=optim,
        grid_dim=grid_dim,
    )

    return amplitude, phase


def raw_data_to_apdistance(sample_data, background_data, target_intensity):
    """
    Convenience function to perform entire processing pipeline:

    1. Subtract background
    2. Clip data to reasonable q range
    3. Denoise with SavGol filter
    4. Smooth and interpolate with Bspline
    5. Scale onto target_intensity
    6. calculate amplitude-phase distance
    """
    raise NotImplementedError
