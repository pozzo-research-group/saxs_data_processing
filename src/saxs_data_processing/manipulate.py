"""
Generic data processing tasks shared between modules

"""
import numpy as np
import pandas as pd

from scipy.interpolate import BSpline, splrep
from scipy.signal import savgol_filter


def ratio_running_average(a, b, n_pts=10):
    """
    Calculate the running average of the ratio of signal to background

    :param a: dataset 1
    :type a: pandas.core.frame.DataFrame
    :param b: dataset 2
    :type b: pandas.core.frame.DataFrame
    :param n_pts: number of points to average over
    :type n_pts: int
    :return running_average: resulting running average of ratio of a/b
    :rtype running_average: np.ndarray
    """

    assert len(a) == len(b), "a and b must have same number of elements"

    ratio = a / b

    running_average = np.convolve(ratio, np.ones(n_pts) / n_pts, mode="same")

    return running_average


def scale_data_integral(data1, data2, scale_qmin, scale_qmax):
    """
    scale data2 onto data1 using a scale factor calculated from the difference in integrals of data1 and data2 over the range scale_qmin -> scale_qmax

    :param data1: SAXS data, scale onto this data
    :type data1: pandas.core.frame.DataFrame
    :param data2: SAXS data, scaled onto other data
    :type data2: pandas.core.frame.DataFrame
    :param scale_qmin: minimum q to integrate over to calculate scaling factor
    :type scale_qmin: float
    :param scale_qmax: maximum q to integrate over to calculate scaling factor
    :type scale_qmax: float
    :return data2_out: data2 scaled to match data1 over q range
    :rtype data2_out: pandas.core.frame.DataFrame
    """

    # get yvals for each dataset
    inrange_data1 = data1[data1["q"].between(scale_qmin, scale_qmax, inclusive="both")]
    inrange_data2 = data2[data2["q"].between(scale_qmin, scale_qmax, inclusive="both")]

    assert np.isclose(inrange_data1["q"].iloc[0], inrange_data2["q"].iloc[0])
    assert np.isclose(inrange_data1["q"].iloc[-1], inrange_data2["q"].iloc[-1])

    x1 = inrange_data1["q"].to_numpy()
    x2 = inrange_data2["q"].to_numpy()
    y1 = inrange_data1["I"].to_numpy()
    y2 = inrange_data2["I"].to_numpy()
    # check for nans in y values

    # trapezoid rule integrate
    scale1 = np.trapz(y1, x1) / (x1[-1] - x1[0])
    scale2 = np.trapz(y2, x2) / (x2[-1] - x2[0])

    scale_factor = scale1 / scale2

    # scale data2 with scale factor
    data2_out = data2.copy()

    data2_out["I"] = data2["I"] * scale_factor

    return data2_out


def interpolate_on_q_linear(target_data, modify_data):
    """
    Linearly interpolate data from modify_data onto q grid from target data

    :param target_data: SAXS data whose q grid is to be matched
    :type target_data: pandas.core.frame.DataFrame
    :param modify_data: data whose q grid needs to be modified to match target_data
    :type modify_data: pandas.core.frame.DataFrame
    :return interp_result: modify data interpolated at q grid points from target data
    :rtype interp_result: pandas.core.frame.DataFrame
    """

    interp_I = np.interp(target_data["q"], modify_data["q"], modify_data["I"])

    interp_result = pd.DataFrame({"q": target_data["q"], "I": interp_I})

    return interp_result


def fit_interpolate_spline(
    q_original,
    I_original,
    q_grid,
    s=0.05,
    k=3,
    clip=True,
    clip_window_loq=10,
    clip_window_hiq=50,
):
    """
    Fit a BSpline to I_original and evaluate it at points on q_grid.

    I_original should be de-noised, for example with a savgol filter, before spline fitting.

    :param q_original: Original, as-measured q vector
    :type q_original: np array
    :param I_original: Denoised intensities at q_original q points
    :type I_original: np array
    :param q_grid: New q points to evaluate spline at
    :type q_grid: np array
    :param s: spline smoothing parameter for scipy splrep (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html). default 0.05
    :type s: float
    :param k: degree of spline fit, 1 <= k <= 5. see more at scipy.splrep docs linked above.
    :type k: int
    :param clip: if True, clip spline blow-ups on extrapolation q points.
    :type clip: Bool
    :param clip_window_loq: Window of original q ponts to evaluate for lo q clip min/max
    :type clip_window: int
    :param clip_window_loq: Window of original q ponts to evaluate for hi q clip min/max
    :type clip_window: int
    :return I_spline: Evaluation of spline fit at q_grid points
    :type I_spline: np array
    """
    assert len(q_original) == len(
        I_original
    ), f"q_original and I_original must have same lengths but have lengths {len(q_original)} and {len(I_original)}"
    spl = BSpline(*splrep(q_original, I_original, s=s, k=k))
    I_spline = spl(q_grid)

    if clip:
        #'cap' spline interp valueas at 1.5x q_original min/max to avoid issues with metric later when spline blows up outside fit range
        I_spline = clip_spline_extrapolation_lowq(
            q_original, I_original, q_grid, I_spline, n_window=clip_window_loq
        )
        I_spline = clip_spline_extrapolation_hiq(
            q_original, I_original, q_grid, I_spline, n_window=clip_window_hiq
        )

    return I_spline


def clip_spline_extrapolation_hiq(
    q_original, I_original, q_grid, I_spline, n_window=50
):
    """
    Clip spline fit blow-ups in hi-q extrapolation range. For spline extrapolation q range, if I_spline is outside max/min of I_original in last n_window q points, set to max/min seen in window.
    """
    # check that q_grid is greater than q_original
    if q_original[-1] > q_grid[-1]:
        return I_spline
    else:
        q_grid_og_max_ind = np.where(q_grid > q_original[-1])[0][0]
        window_max = np.max(I_original[-n_window:])
        window_min = np.min(I_original[-n_window:])

        I_spline_extrap = I_spline[q_grid_og_max_ind:]

        I_spline_extrap[np.where(I_spline_extrap > window_max)] = window_max
        I_spline_extrap[np.where(I_spline_extrap < window_min)] = window_min

        I_spline_clip = I_spline
        I_spline_clip[q_grid_og_max_ind:] = I_spline_extrap

        return I_spline_clip


def clip_spline_extrapolation_lowq(
    q_original, I_original, q_grid, I_spline, n_window=10
):
    """
    Clip spline fit blow-ups in hi-q extrapolation range. For spline extrapolation q range, if I_spline is outside max/min of I_original in last n_window q points, set to max/min seen in window.
    """
    # check that q_grid is greater than q_original
    if q_original[0] < q_grid[0]:
        return I_spline
    else:
        q_grid_og_min_ind = np.where(q_grid < q_original[0])[0][-1]
        window_max = np.max(I_original[:n_window])
        window_min = np.min(I_original[:n_window])

        print(window_max)
        print(window_min)
        print(q_grid_og_min_ind)
        I_spline_extrap = I_spline[:q_grid_og_min_ind]

        I_spline_extrap[np.where(I_spline_extrap > window_max)] = window_max
        I_spline_extrap[np.where(I_spline_extrap < window_min)] = window_min

        I_spline_clip = I_spline
        I_spline_clip[:q_grid_og_min_ind] = I_spline_extrap

        return I_spline_clip


def scale_intensity_highqavg(I_measured, I_target, n_avg):
    """
    Scales I_measured onto I_target based on the average value of the last n_avg data points in both.

    Pass intensities that are already in log10 space. Scales by taking average value of I_measured and I_target in 10**(I) space, then multiplying 10**I by their ratio, then returning log((10**I)*ratio).
    :param I_measured: Measured intensity to be scaled,  in log10 space
    :type I_measured: np array
    :param I_target: target intensity to scale I_measured onto, in log10 space
    :type I_target: np array
    :param n_avg: the number of high-q data points to incorporate into scaling average calculation
    :return I_scaled: Scaled I_measured, in log10 space
    :rtype I_scaled: np array
    """
    # 1. convert both back into I space from logI space
    pow_meas = 10**I_measured
    pow_target = 10**I_target
    # 2. get quotient
    avg_target = np.mean(pow_target[-n_avg:])
    avg_meas = np.mean(pow_meas[-n_avg:])
    quotient = np.abs(avg_target / avg_meas)
    # 3. scale measured
    scaled_meas = pow_meas * quotient
    # convert back to log space
    I_scaled = np.log10(scaled_meas)
    return I_scaled

    return


def denoise_intensity(I_measured, savgol_n=15, savgol_order=3):
    """
    Apply Savitzky-Golay smooothing filter to I_measured.

    Scipy SavGol docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    :param I_measured: Intensity data to apply filter to
    :type I_measured: np array
    :param savgol_n: length of savgol filter, default 15
    :type savgol_n: int
    :param savgol_order: Order of polynomials to use in filter. Default 3
    :type savgol_order: int
    :return I_savgol: Filtered intensity
    :rtype I_savgol: np array
    """
    I_savgol = savgol_filter(I_measured, savgol_n, savgol_order)

    return I_savgol
