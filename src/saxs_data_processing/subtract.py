import numpy as np
from saxs_data_processing import manipulate
import warnings


def find_ratio_peak(ratio_avg):
    """
    Find max value of ratio using max
    Returns index of peak. Handles nans and infs somewhat gracefully

    :param ratio_avg: data to find the max of
    :type ratio_avg: pandas.core.frame.DataFrame
    :return ind: index of max value in data
    :rtype ind: int
    """

    avg_copy = ratio_avg.copy()
    avg_copy[avg_copy == np.inf] = 0
    ind = np.nanargmax(avg_copy)

    return ind


def subtract_background(
    data, background, scale_background=False, scale_qmin=1e-4, scale_qmax=1e-3
):
    """
    Subtract background from data

    Assumes that data and background are both measurd on the same q grid to within a tight tolerance. If this is not true, you must inerpolate your data onto the same grid first.

    :param data: SAXS data from sample
    :type data: pandas.core.frame.DataFrame
    :param background: SAXS data from background
    :type background: pandas.core.frame.DataFrame
    :param scale_background: Option to scale background onto data
    :type scale_background: bool
    :param scale_qmin: lower limit of q range used for scaling. Only used if scale_background True
    :type scale_qmin: float
    :param scale_qmax: uppper limit of q range used for scaling. Only used if scale_background True
    :type scale_qmax: float
    :return subtracted_data: resulting data from data - background
    :rtype subtracted_data: pandas.core.frame.DataFrame

    """
    # Get overlapping q range and only subtract/return data from here
    data_inclusive = data[data["q"].isin(background["q"])]
    background_inclusive = background[background["q"].isin(data["q"])]

    # check that q values line up for everything
    assert np.isclose(
        data_inclusive["q"].to_numpy(), background_inclusive["q"].to_numpy()
    ).all()
    # once we get fancier look into allowance for slop or interpolation options. For now throw out anything not compliant ^^

    if scale_background:
        background_inclusive = manipulate.scale_data(
            data_inclusive, background_inclusive, scale_qmin, scale_qmax
        )

    subtracted_I = data_inclusive["I"] - background_inclusive["I"]

    subtracted_data = data_inclusive.copy()

    subtracted_data["I"] = subtracted_I

    return subtracted_data


def select_valid_data(signal, background, lowq_thresh=5, hiq_thresh=5, hiq_avg_pts=10):
    """
    Find the region of valid data in SAXS signal

    Considers the ratio of signal to background to identify regions with enough scattering to provide information. Threshold for valid data deterined by user.

    :param signal: signal saxs data
    :type signal: pandas.core.frame.DataFrame
    :param background: corresponding background
    :type background: pandas.core.frame.DataFrame
    :param lowq_thresh: multiplier threshold for low q limit. Signal must be low_thresh times larger than background to find low q limit
    :type lowq_thresh: float
    :param hiq_thresh: threshold for hiq limit
    :type hiq_thresh: float
    :param hiq_avg_pts: how many data points to take a running average over when considering thresholds
    :type hiq_avg_pts: int
    :return lowq_lim: count index of low-q data limit (ie data beyond the lowq_lim'th row is good)
    :rtype loq_lim: int
    :return hiq_lim: count index of hi-q data limit
    :rtype hiq_lim: int
    """

    assert len(signal) == len(
        background
    ), "Signal and background data sets need to have same number of data points"

    # TODO: check that signal and background are on the same grid

    q = signal["q"]

    lowq_lim = None
    hiq_lim = None

    last_n_ratios = []
    rolling_average_ratio = manipulate.ratio_running_average(
        signal["I"], background["I"], n_pts=hiq_avg_pts
    )

    # need to find peak to engage hi-q limit finder
    ratio_peak_ind = find_ratio_peak(rolling_average_ratio)

    for i, ratio in enumerate(rolling_average_ratio):
        if ratio > lowq_thresh:
            if lowq_lim is None:
                lowq_lim = i

        if lowq_lim is not None:
            if i > ratio_peak_ind:
                if ratio < hiq_thresh:
                    if hiq_lim is None:
                        hiq_lim = i
                        break

    hiq_lim = i
    warnings.warn("No hiq lim satisfying threshold found, setting value to q max")

    if lowq_lim is None:
        warnings.warn(
            "Failed to find region of valid data (low q limit not found). Check that your sample scatters reasonably well"
        )
        return lowq_lim, hiq_lim
    if hiq_lim is None:
        warnings.warn(
            "Failed to find region of valid data (low q limit not found). Check that your sample scatters reasonably well"
        )
        return lowq_lim, hiq_lim
    if hiq_lim - lowq_lim < 30:
        warnings.warn(
            "Insufficient data points in q range with scattering. Check data quality"
        )

    return lowq_lim, hiq_lim


def chop_subtract(
    signal,
    background,
    lowq_thresh=5,
    hiq_thresh=5,
    hiq_avg_pts=10,
    scale=False,
    low_q_hard_merge=None,
    hi_q_hard_merge=None,
    hi_q_cutoff=True,
):
    """
    Background subtract and select valid data region for unprocessed 1D SAXS data

    :param signal: SAXS signal data
    :type signal: pandas.core.frame.DataFrame
    :param background: SAXS background data
    :type background: pandas.core.frame.DataFrame
    :param lowq_thresh: ratio criteria for low q data validity. Signal must be lowq_thresh times larger than background for data to be kept
    :type lowq_thresh: float
    :param hiq_thresh: ratio criteria for hi q data validity. Signal must be hiq_thresh times larger than backbround for data to be kept
    :type hiq_thresh: int
    :param hiq_avg_pts: Number of points to average over for noise reduction for hi-q threshold
    :type hiq_avg_pts: int
    :param scale: whether or not to scale background onto signal during subtract
    :type scale: bool
    :param low_q_hard_merge: None - use threshold based valid data selection, int - use this index as low q data validity limit
    :type low_q_hard_merge: None, int
    :param hi_q_hard_merge: None - use threshold based valid data selection, int - use this as hi q data validity limit
    :type hi_q_hard_merge: None, int
    :param hi_q_cutoff: whether or not to chop off hi q data based on ratio limit
    :param hi_q_cutoff: bool
    :return chopped_subtracted: 1D saxs data, background subtracted and chopped to valid data q range
    :rtype chopped_subtracted: pandas.core.frame.DataFrame
    """

    # Get overlapping q range and only subtract/return data from here
    signal = signal[signal["q"].isin(background["q"])]
    background = background[background["q"].isin(signal["q"])]
    assert np.isclose(signal["q"].to_numpy(), background["q"].to_numpy()).all()

    # if user supplied both limits, no need to run auto find. else do run this
    if low_q_hard_merge is not None and hi_q_hard_merge is not None:
        loq = low_q_hard_merge
        hiq = hi_q_hard_merge

    else:
        loq, hiq = select_valid_data(
            signal,
            background,
            lowq_thresh=lowq_thresh,
            hiq_thresh=hiq_thresh,
            hiq_avg_pts=hiq_avg_pts,
        )
    # print(f'lo q: {loq}, hi q: {hiq}')
    if (loq == None) or (hiq == None):
        warnings.warn("Issue during data selection, check data quality")
        return None

    # still need to override valid data range with user suppplied points if provided
    if low_q_hard_merge is not None:
        loq = low_q_hard_merge
    if hi_q_hard_merge is not None:
        hiq = hi_q_hard_merge

    subtracted_signal = subtract_background(signal, background, scale_background=scale)

    if not hi_q_cutoff:
        hiq = -1

    chopped_subtracted = subtracted_signal.iloc[loq:hiq].copy()

    return chopped_subtracted
