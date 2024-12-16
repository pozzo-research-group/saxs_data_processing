import numpy as np
import pandas as pd
from saxs_data_processing import manipulate


def splice_datasets(
    low_q_data, hi_q_data, low_q_limit, hi_q_limit, low_q_source=None, hi_q_source=None
):
    """
    Splice together low_q_data and hi_q_data.

    At q<lo_q_limit, only low_q_data included. Above hi_q_limit, only hi_q_data included. Between lo_q and hi_q limits, both datasets included

    :param low_q_data: Dataset to keep low q data for
    :type low_q_data: pandas.core.frame.DataFrame
    :param hi_q_data: Dataset to keep high q data for
    :type hi_q_data: pandas.core.frame.DataFrame
    :param low_q_limit: q value for low q cutoff
    :type low_q_limit: float
    :param hi_q_limit: q value for hi q cutoff
    :type hi_q_limit: float
    :param low_q_source: Dataset name for low q data
    :type loq_q_source: str, None
    :param hi_q_source: Dataset name for hi q data
    :type hi_q_source: str, None
    :return spliced_data: Combined dataset
    :rtype spliced_data: pandas.core.frame.DataFrame
    """
    low_q_include = low_q_data[low_q_data["q"] < hi_q_limit].copy()
    hi_q_include = hi_q_data[hi_q_data["q"] > low_q_limit].copy()

    if low_q_source is not None:
        low_q_include["source"] = low_q_source
    if hi_q_source is not None:
        hi_q_include["source"] = hi_q_source

    spliced_data = pd.concat([low_q_include, hi_q_include]).sort_values("q")

    return spliced_data


def find_overlap(data1, data2):
    """
    Returns the subset of data1 that has q values strictly enclosed by q range of data2

    :param data1: SAXS data
    :type data1: pandas.core.frame.DataFrame
    :param data2: SAXS data
    :type data2: pandas.core.frame.DataFrame
    :return data1_overlap: subset of data1 with q values within q range of data2
    :rtype data1_overlap: pandas.core.frame.DataFrame
    """

    # Find the range of data1 that is entirely within data2

    data1_qmax = data1["q"].max()
    data1_qmin = data1["q"].min()

    data2_qmax = data2["q"].max()
    data2_qmin = data2["q"].min()

    data1_overlap = data1[data1["q"] > data2_qmin]
    data1_overlap = data1_overlap[data1_overlap["q"] < data2_qmax]

    return data1_overlap


def forward_difference(data):
    """
    Finite first order forward differential

    :param data: SAXS data
    :type data: pandas.core.frame.DataFrame
    :return data: Data with 1st derivative added as 'dIdq' column
    :rtype data: pandas.core.frame.DataFrame
    """

    q = data["q"]
    I = data["I"]

    dI = np.diff(I)
    dq = np.diff(q)

    dIdq = dI / dq

    data = data[:-1].copy()
    data["dIdq"] = dIdq

    return data


def deriv_ratio(data1, data2):
    """Compute the ratio of the 1st derivatives of 2 datasets

    Datasets should have same length and q grid

    :param data1: SAXS data
    :type data1: pandas.core.frame.DataFrame
    :param data2: SAXS data
    :type data2: pandas.core.frame.DataFrame
    :return ratio: Ratio of data1 derivative over data2 derivative
    :rtype ratio: Pandas series?

    """
    deriv1 = forward_difference(data1)
    deriv2 = forward_difference(data2)

    ratio = deriv1["dIdq"] / deriv2["dIdq"]

    return ratio


def noise_score(data, n_pts=20):
    """Calculate a measure of noisiness in data by looking at ratio of data to n_pts local running average (backwards average here I think)

    :param data: SAXS data
    :type data: pandas.core.frame.DataFrame
    :param n_pts: number of points to use for running average
    :type n_pts: int
    :return noise_score: measure of noisiness of data
    :rtype noise_score: series?
    """
    running_average = np.convolve(data["I"], np.ones(n_pts) / n_pts, mode="same")

    ratio_to_average = data["I"] / running_average

    noise_score = abs(1 - ratio_to_average)

    return noise_score


def find_qlim_low(low_q_data, hi_q_data, val_threshold=0.1, slope_threshold=0.4):
    """
    Find the low q merge limit using combination of curve closeness and slope criteria

    :param low_q_data: low q SAXS data
    :type low_q_data: pandas.core.frame.DataFrame
    :param hi_q_data: hi q SAXS data
    :type hi_q_data: pandas.core.frame.DataFrame
    :param val_threshold: threshold for the ratio of low q data to hi q data for low q limit to be met
    :type val_threshold: float
    :param slope_threshold: threshold for the ratio of slopes of low_q_data and hi_q_data to be close for low q limit to be met
    :type slope_threshold: float
    :return q: q value for low q criteria to be satisfied
    :type q: float
    """
    slope_ratio = deriv_ratio(low_q_data, hi_q_data)
    val_ratio = manipulate.ratio_running_average(low_q_data["I"], hi_q_data["I"])

    match_count = 0

    for q, val, slope in zip(low_q_data["q"][1:], val_ratio[1:], slope_ratio):

        if abs(1 - val) < val_threshold:
            if abs(1 - slope) < slope_threshold:
                match_count += 1

        if match_count == 3:
            # we've found limit
            return q

    raise AssertionError("Q low limit matching given criteria not found")


def find_qlim_hi(low_q_data, qlim_low, noise_threshold=0.2, n_pts=20):
    """
    Find the hi q merge limit using noise criteria on low q data

    :param low_q_data: low q SAXS data
    :type low_q_data: pandas.core.frame.DataFrame
    :param hi_q_data: hi q SAXS data
    :type hi_q_data: pandas.core.frame.DataFrame
    :param noise_threshold: threhold for amount of noise to tolerate before tripping hi q limit
    :type noise_threshold: float
    :param n_pts: number of points to average over for noise calculation
    :type n_pts: int
    :retun q: hi q limit
    :type q: float
    """

    noise_value = noise_score(low_q_data, n_pts=n_pts)

    for i, (q, noise) in enumerate(zip(low_q_data["q"], noise_value)):
        if i > n_pts:  # need to avoid edge effects
            if q > qlim_low:
                if noise > noise_threshold:
                    return q

    raise AssertionError("Q lim hi not found with given criteria")


def get_merge_limits(
    low_q_data,
    hi_q_data,
    val_threshold=0.1,
    slope_threshold=0.4,
    noise_threshold=0.2,
    n_pts=20,
):
    """Find low q and hi q merge limits for data using criteria

    :param low_q_data: low q SAXS data
    :type low_q_data: pandas.core.frame.DataFrame
    :param hi_q_data: hi q SAXS data
    :type hi_q_data: pandas.core.frame.DataFrame
    :param val_threshold: threshold for the ratio of low q data to hi q data for low q limit to be met
    :type val_threshold: float
    :param slope_threshold: threshold for the ratio of slopes of low_q_data and hi_q_data to be close for low q limit to be met
    :type slope_threshold: float
    :param noise_threshold: threhold for amount of noise to tolerate before tripping hi q limit
    :type noise_threshold: float
    :param n_pts: number of points to average over for noise calculation
    :type n_pts: int
    :return (qlim_low, qlim_hi): Tuple of low and hi q limits as floats
    :rtype (qlim_low, qlim_hi): (float, float)

    """

    qlim_low = find_qlim_low(
        low_q_data,
        hi_q_data,
        val_threshold=val_threshold,
        slope_threshold=slope_threshold,
    )
    qlim_hi = find_qlim_hi(
        low_q_data, qlim_low, noise_threshold=noise_threshold, n_pts=n_pts
    )

    return (qlim_low, qlim_hi)


def auto_merge(
    low_q_data,
    hi_q_data,
    low_q_source=None,
    hi_q_source=None,
    val_threshold=0.2,
    slope_threshold=0.4,
    noise_threshold=0.25,
    n_pts=20,
    low_q_hard_merge=None,
    hi_q_hard_merge=None,
):
    """
    Merge low_q_data with hi_q_data. Select merge points using criteria specified by user


    :param low_q_data: low q SAXS data
    :type low_q_data: pandas.core.frame.DataFrame
    :param hi_q_data: hi q SAXS data
    :type hi_q_data: pandas.core.frame.DataFrame
    :param low_q_source: Dataset name for low q data source
    :type low_q_source: str
    :param hi_q_source: Dataset name for hi q data source
    :type hi_q_source: str
    :param val_threshold: threshold for the ratio of low q data to hi q data for low q limit to be met
    :type val_threshold: float
    :param slope_threshold: threshold for the ratio of slopes of low_q_data and hi_q_data to be close for low q limit to be met
    :type slope_threshold: float
    :param noise_threshold: threhold for amount of noise to tolerate before tripping hi q limit
    :type noise_threshold: float
    :param n_pts: number of points to average over for noise calculation
    :type n_pts: int
    :param low_q_hard_merge: Set to a q value to override automatic merge region calculations
    :type low_q_hard_merge: None, float
    :param hi_q_hard_merge: set to a q value to override automatic merge region calculation
    :type hi_q_hard_merge: None, float
    :return spliced: Dataset spliced together from low q and hi q data at merge limits
    :rtype spliced: pandas.core.frame.DataFrame
    :param merge_metadata: Metadata from merge process with data sources and merge parameters
    :rtype merge_metadata: dict

    Takes subtracted/chopped data
    """

    low_q_overlap = find_overlap(low_q_data, hi_q_data)
    hi_q_overlap = find_overlap(hi_q_data, low_q_data)

    hi_q_interpolated = manipulate.interpolate_on_q(low_q_overlap, hi_q_overlap)

    if low_q_hard_merge is not None and hi_q_hard_merge is not None:
        low_q_lim = low_q_hard_merge
        hi_q_lim = hi_q_hard_merge
    else:
        low_q_lim, hi_q_lim = get_merge_limits(
            low_q_overlap,
            hi_q_interpolated,
            val_threshold=val_threshold,
            slope_threshold=slope_threshold,
            noise_threshold=noise_threshold,
            n_pts=n_pts,
        )

    if low_q_hard_merge is not None:
        low_q_lim = low_q_hard_merge
    if hi_q_hard_merge is not None:
        hi_q_lim = hi_q_hard_merge

    print(f"low q merge lim: {low_q_lim}")
    print(f"hi q merge lim: {hi_q_lim}")

    spliced = splice_datasets(
        low_q_data, hi_q_data, low_q_lim, hi_q_lim, low_q_source, hi_q_source
    )

    merge_metadata = {
        "low_q_source": low_q_source,
        "hi_q_source": hi_q_source,
        "low_q_limit": low_q_lim,
        "hi_q_limit": hi_q_lim,
        "low_q_value_threshold": val_threshold,
        "low_q_slope_threshold": slope_threshold,
        "hi_q_noise_threshold": noise_threshold,
        "averaging_n_pts": n_pts,
    }

    return spliced, merge_metadata
