import numpy as np
import manipulate

def find_ratio_peak(ratio_avg):
    """
    Find max value of ratio using max
    Returns index of peak
    """
    
    avg_copy = ratio_avg.copy()
    avg_copy[avg_copy == np.inf] = 0
    ind = np.nanargmax(avg_copy)
    
    return ind


def subtract_background(data, background, scale_background = False, scale_qmin = 1e-4, scale_qmax = 1e-3):

    # check that q values line up for everything
    assert np.isclose(data['q'].to_numpy(), background['q'].to_numpy()).all()
    # once we get fancier look into allowance for slop or interpolation options. For now throw out anything not compliant ^^
    
    if scale_background:
        background = manipulate.scale_data(data, background, scale_qmin, scale_qmax)
        
    subtracted_I = data['I'] - background['I']

    subtracted_data = data.copy()

    subtracted_data['I'] = subtracted_I
    
    return subtracted_data


def select_valid_data(signal, background, lowq_thresh = 5, hiq_thresh = 5, hiq_avg_pts = 10):
    """
    Find the region of valid data in SAXS signal
    
    Considers the ratio of signal to background to identify regions with enough scattering to provide information
    
    Parameters:
    -----------
    signal (DataFrame): signal saxs data
    background (DataFrame): corresponding background
    lowq_thresh: multiplier threshold for low q limit. Signal must be low_thresh times larger than background to find low q limit
    hiq_thresh: threshold for hiq limit
    hiq_avg_pts: how many data points to take a running average over when considering thresholds.
    """

    assert len(signal) == len(background), 'Signal and background data sets need to have same number of data points'

    q = signal['q']

    lowq_lim = None
    hiq_lim = None

    last_n_ratios = []
    rolling_average_ratio = manipulate.ratio_running_average(signal['I'], background['I'], n_pts = hiq_avg_pts)

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
                        
    if lowq_lim is None:
        raise AssertionError('Failed to find region of valid data (low q limit not found). Check that your sample scatters reasonably well')
    if hiq_lim is None:
        raise AssertionError('Failed to find region of valid data (low q limit not found). Check that your sample scatters reasonably well')
        
    if hiq_lim - lowq_lim < 30:
        raise AssertionError('Insufficient data points in q range with scattering. Check data quality')

    return lowq_lim, hiq_lim


def chop_subtract(signal, background, lowq_thresh = 5, hiq_thresh = 5, hiq_avg_pts = 10, scale = False):
    """
    wrapper function to handle subtraction and valid data selection 
    """
    assert np.isclose(signal['q'].to_numpy(), background['q'].to_numpy()).all()

    loq, hiq = select_valid_data(signal, background, lowq_thresh = lowq_thresh, hiq_thresh= hiq_thresh, hiq_avg_pts = hiq_avg_pts)

    subtracted_signal = subtract_background(signal, background, scale_background = False)

    chopped_subtracted = subtracted_signal.iloc[loq:hiq]

    return chopped_subtracted

