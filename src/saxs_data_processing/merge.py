import numpy as np
import pandas as pd
import manipulate

def splice_datasets(low_q_data, hi_q_data, low_q_limit, hi_q_limit, low_q_source = None, hi_q_source = None):
    """
    Splice together low_q_data and hi_q_data. At q<lo_q_limit, only low_q_data included. Above hi_q_limit, only hi_q_data included. Between lo_q and hi_q limits, both datasets included
    """
    low_q_include = low_q_data[low_q_data['q'] < hi_q_limit].copy()
    hi_q_include = hi_q_data[hi_q_data['q'] > low_q_limit].copy()

    if low_q_source is not None:
        low_q_include['source'] = low_q_source
    if hi_q_source is not None:
        hi_q_include['source'] = hi_q_source

    spliced_data = pd.concat([low_q_include, hi_q_include]).sort_values('q')

    return spliced_data

def find_overlap(data1, data2):
    """
    Returns the subset of data1 that has q values strictly enclosed by q range of data2
    """

    # Find the range of data1 that is entirely within data2

    data1_qmax = data1['q'].max()
    data1_qmin = data1['q'].min()

    data2_qmax = data2['q'].max()
    data2_qmin = data2['q'].min()

    data1_overlap = data1[data1['q'] > data2_qmin]
    data1_overlap = data1_overlap[data1_overlap['q'] < data2_qmax]

    return data1_overlap

def forward_difference(data):


    q = data['q']
    I = data['I']

    dI = np.diff(I)
    dq = np.diff(q)

    dIdq = dI/dq
    
    data = data[:-1].copy()
    data['dIdq'] = dIdq
    
    return data

def deriv_ratio(data1, data2):
    deriv1 = forward_difference(data1)
    deriv2 = forward_difference(data2)
    
    ratio = deriv1['dIdq']/deriv2['dIdq']
    
    return ratio


def noise_score(data, n_pts = 20):
    """
    Calculate a measure of noisiness in data by looking at ratio of data to n_pts local running average (backwards average here I think)
    """
    running_average = np.convolve(data['I'], np.ones(n_pts)/n_pts, mode = 'same')

    ratio_to_average = data['I']/running_average

    noise_score = abs(1-ratio_to_average)
    
    return noise_score 

def find_qlim_low(low_q_data, hi_q_data, val_threshold = 0.1, slope_threshold = 0.4):
    """
    Find the low q merge limit using combination of curve closeness and slope criteria 
    """
    slope_ratio = deriv_ratio(low_q_data, hi_q_data)
    val_ratio = manipulate.ratio_running_average(low_q_data['I'], hi_q_data['I'])

    match_count = 0

    for q, val, slope in zip(low_q_data['q'][1:], val_ratio[1:], slope_ratio):

        if abs(1-val) < val_threshold:
            if abs(1-slope) < slope_threshold:
                match_count +=1

        if match_count == 3:
            # we've found limit
            return q
        
    raise AssertionError('Q low limit matching given criteria not found')


def find_qlim_hi(low_q_data, qlim_low, noise_threshold = 0.2, n_pts = 20):
    """
    Find the hi q merge limit using noise criteria on low q data 
    """
    
    noise_value = noise_score(low_q_data, n_pts = n_pts)

    for i, (q, noise) in enumerate(zip(low_q_data['q'], noise_value)):
        if i > n_pts: # need to avoid edge effects 
            if q > qlim_low:
                if noise > noise_threshold:
                    return q
                
    raise AssertionError('Q lim hi not found with given criteria')


def get_merge_limits(low_q_data, hi_q_data, val_threshold = 0.1, slope_threshold = 0.4, noise_threshold = 0.2, n_pts = 20):
    
    qlim_low = find_qlim_low(low_q_data, hi_q_data, val_threshold = val_threshold, slope_threshold = slope_threshold)
    qlim_hi = find_qlim_hi(low_q_data, qlim_low, noise_threshold = noise_threshold, n_pts = n_pts)
    
    return (qlim_low, qlim_hi)


def auto_merge(low_q_data, hi_q_data, low_q_source = None, hi_q_source = None, val_threshold = 0.2, slope_threshold = 0.4, noise_threshold = 0.25, n_pts = 20):
    """
    Wrapper function to handle merging tasks 
    
    Takes subtracted/chopped data 
    """
    
    low_q_overlap = find_overlap(low_q_data, hi_q_data)
    hi_q_overlap = find_overlap(hi_q_data, low_q_data)
    
    hi_q_interpolated = manipulate.interpolate_on_q(low_q_overlap, hi_q_overlap)
    
    low_q_lim, hi_q_lim = get_merge_limits(low_q_overlap, hi_q_interpolated, val_threshold = val_threshold, slope_threshold = slope_threshold, noise_threshold = noise_threshold, n_pts = n_pts)
    
    print(f'low q merge lim: {low_q_lim}')
    print(f'hi q merge lim: {hi_q_lim}')
    
    spliced = splice_datasets(low_q_overlap, hi_q_overlap, low_q_lim, hi_q_lim, low_q_source, hi_q_source)
    
    merge_metadata = {'low_q_source':low_q_source, 'hi_q_source':hi_q_source, 'low_q_limit':low_q_lim, 'hi_q_limit':hi_q_lim, 'low_q_value_threshold':val_threshold, 'low_q_slope_threshold':slope_threshold, 'hi_q_noise_threshold':noise_threshold, 'averaging_n_pts':n_pts}
    
    return spliced, merge_metadata