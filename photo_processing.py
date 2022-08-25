'''Contains functions for analyzing CHILLAX photos

This set of functions is for processing gray-scaled photos in numpy array form.
The photos are consistent in their dimensions and their view of the CHILLAX detector.

'''

import numpy as np

def slice_photo(A, num_slices=200, slice_length=450, slice_start_x=950, slice_start_y=2150):
    '''Find average of diagonal slices of pixels perpendicular to 
    ring of ice in detector can.
    
    Input:
    A -- 2D array representing gray-scaled photo
    num_slices -- number of adjacent slices to average over
    slice_length -- length of a slice in pixels
    slice_start_x -- x coordinate of first pixel in first slice
    slice_start_y -- y coordinate of first pixel in first slice
    
    Output:
    photo_slice -- 1D array representing pixel brightness along average slice
    '''
    
    photo_slice = np.zeros(slice_length)
    for i in range(num_slices):
        for j in range(slice_length):
            photo_slice[j] += A[slice_start_x+j-int((i+1)/2),slice_start_y+int(i/2)+j] / 200
    return photo_slice

def rolling_average(x, half_window):
    '''Smooth set of waveforms with rolling average
    
    Inputs:
    x -- 2D array of waveforms
    half_window -- rolling average at position n is taken from elements half_window before to half_window after n
    
    Output:
    2D array of waveforms with same shape as x, smoothed by rolling average
    '''
    
    cumsum = np.cumsum(np.insert(x,0,0,axis=1),axis=1)
    averaged = (cumsum[:,2*half_window+1:] - cumsum[:,:-(2*half_window+1)]) / float(2*half_window+1)
    averaged_left = np.zeros((len(x),half_window))
    averaged_right = np.zeros((len(x),half_window))
    averaged_left[:, 0:half_window] = np.broadcast_to(averaged[:,0], (half_window,len(x))).T
    averaged_right[:, 0:half_window] = np.broadcast_to(averaged[:,-1],(half_window,len(x))).T
    return np.append(np.append(averaged_left,averaged,axis=1),averaged_right,axis=1)

def leading_baseline(waveform, trigger_sample, start_sample, pretr_excl_sampl, linear=False):
    """Find baseline and standard deviation at beginning of waveform

    Find the baseline for a numpy array of waveforms where each row is one
    waveform. Uses the given number of leading samples to determine the
    mean and standard deviation, which are returned as a tuple. This is a
    pretty basic way to find the baseline.

    Inputs:
    waveform -- 2D numpy array where each row is a waveform
    trigger_sample -- integer indicating rough expected start of pulse (used to determine what part of waveform to ignore for baseline)
    start_sample -- integer giving index to start baseline calculation
    pretr_excl_sampl -- integer giving num of pretrigger samples to exclude
    
    Output:
    Mean, standard deviation of leading samples of waveform
    """
    
    samples = np.arange(start_sample,trigger_sample-pretr_excl_sampl)
    if linear:
        baseline = np.zeros(np.shape(waveform))
        std = np.zeros(len(waveform))
        for i in range(len(waveform)):            
            fit_params, fit_cov = scipy.optimize.curve_fit(lin_fit,samples,waveform[i,start_sample:trigger_sample-pretr_excl_sampl])
            baseline[i] = lin_fit(np.arange(len(waveform[i])),*fit_params)
            std[i] = np.std(waveform[i,start_sample:trigger_sample-pretr_excl_sampl]-baseline[i,start_sample:trigger_sample-pretr_excl_sampl])
        return baseline, std
    else:
        return (np.mean(waveform[:,start_sample:trigger_sample-pretr_excl_sampl],
                        axis = 1),
                np.std(waveform[:,start_sample:trigger_sample-pretr_excl_sampl],
                        axis = 1))
    
def std_dev_pulsefinding(waveform, std_dev, trigger_sample, search_window, frame=5,
                        rising_thresh=5.0, falling_thresh=3.0):
    """Finds a pulse given a baseline standard deviation

    Given a 1D numpy array representing a waveform and the standard deviation
    of its baseline, returns the starting and ending index of the first pulse
    in the waveform. Note this functionality is different from other functions
    in this module since it does not work for the whole numpy array of
    waveforms

    Inputs:
    waveform -- 1D numpy array corresponding to a waveform
    std_dev -- float giving standard deviation of baseline
    trigger_sample -- integer indicating rough expected start of pulse (used to determine what part of waveform to ignore for baseline)
    search_window -- search for pulses within search_window before and after trigger sample
    rising_thresh -- float giving number std devs to start pulse (def. = 5.0)
    falling_thresh -- float giving number std devs to end pulse (def. = 3.0)
    
    Output:
    start, end indices of pulse
    """
    
    window_start = trigger_sample - search_window
    window_end = trigger_sample + search_window

    pulse_window = waveform[window_start:window_end]

    pulse_start = np.argmax(pulse_window > rising_thresh*std_dev) + window_start

    # Pulse ends one sample after starting index at earliest
    pulse_end = np.argmax(waveform[pulse_start+1:] < falling_thresh*std_dev) + pulse_start + 1
    if max(waveform[pulse_start+1:] < falling_thresh*std_dev)==0:
        pulse_end = len(waveform)-1

    return pulse_start, pulse_end