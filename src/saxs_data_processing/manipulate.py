"""
Generic data processing tasks shared between modules

"""
import numpy as np
import pandas as pd

def ratio_running_average(a, b, n_pts = 10):
    """
    Calculate the running average of the ratio of signal to background
    
    a - signal data. 1d array
    b - background data. 1d array
    """
    
    assert len(a) == len(b), 'a and b must have same number of elements'
    
    ratio = a/b
    
    running_average = np.convolve(ratio, np.ones(n_pts)/n_pts, mode = 'same')
    
    return running_average


def scale_data(data1, data2, scale_qmin, scale_qmax):
    """
    scale data2 onto data1 using a scale factor calculated from the difference in integrals of data1 and data2 over the range scale_qmin -> scale_qmax
    """

    # get yvals for each dataset
    inrange_data1 = data1[data1['q'].between(scale_qmin, scale_qmax, inclusive = 'both')]
    inrange_data2 = data2[data2['q'].between(scale_qmin, scale_qmax, inclusive = 'both')]

    assert np.isclose(inrange_data1['q'].iloc[0], inrange_data2['q'].iloc[0])
    assert np.isclose(inrange_data1['q'].iloc[-1], inrange_data2['q'].iloc[-1])


    x1 = inrange_data1['q'].to_numpy()
    x2 = inrange_data2['q'].to_numpy()
    y1 = inrange_data1['I'].to_numpy()
    y2 = inrange_data2['I'].to_numpy()
    #check for nans in y values

    #trapezoid rule integrate
    scale1 = np.trapz(y1, x1)/(x1[-1] - x1[0])
    scale2 = np.trapz(y2, x2)/(x2[-1] - x2[0])

    scale_factor = scale1/scale2


    # scale data2 with scale factor
    data2_out = data2.copy()

    data2_out['I'] = data2['I']*scale_factor

    return data2_out

def interpolate_on_q(target_data, modify_data):
    """
    Linearly interpolate data from modify_data onto q grid from target data 
    """

    interp_I = np.interp(target_data['q'], modify_data['q'], modify_data['I'])

    interp_result = pd.DataFrame({'q':target_data['q'], 'I':interp_I})
    
    return interp_result