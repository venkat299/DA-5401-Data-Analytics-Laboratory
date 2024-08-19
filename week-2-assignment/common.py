import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_assignment_data():
    _dir = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(_dir, './Assignment2.data'), sep='\t')        

# OLS : estimate the value of the beta vector assuming that X is made of independent features.
def estimateBeta(X, y):
    numerator = np.matmul(np.transpose(X), y)
    denom = np.matmul(np.transpose(X), X)
    denom_inv = np.linalg.inv(denom)
    beta = np.matmul(denom_inv, numerator)
    return beta

# create a helper that would estimate yhat from X and beta.
def predict(beta, X):
    # reshape the input to a matrix, if it is appearing like an 1d array.
    if len(X.shape) != 2:
        X = np.expand_dims(X,1)
    # convert the beta list in to an array.
    beta = np.array(beta)
    # perform estimation of yhat.
    return np.matmul(X, beta)

# compute the sum of squared error between y and yhat.
def SSE(y, yhat):
    return np.sum((y-yhat)**2)

def MSE(y, yhat):
    return np.sum((y-yhat)**2)/y.size



def find_zero_crossings(data, threshold=0.0):
    """
    Finds the zero crossings in a given data array.
    
    Parameters:
    data (numpy array): The input data array representing the damped oscillation.
    threshold (float): The value considered as zero for crossing detection.
    
    Returns:
    numpy array: Indices of the zero crossings in the data array.
    """
    zero_crossings = []
    
    # Loop through data and detect zero crossings
    for i in range(1, len(data)):
        if (data[i-1] > threshold and data[i] <= threshold) or (data[i-1] < threshold and data[i] >= threshold):
            zero_crossings.append(i)
    
    return np.array(zero_crossings)



def find_phase_from_zero_crossings(amplitudes, omega, zero_crossings):
    """
    Estimate the phase of a damped oscillation given the angular frequency and number of zero crossings.

    Parameters:
    - amplitudes: Array-like, amplitude measurements.
    - omega: Angular frequency.
    - zero_crossings: Number of zero crossings observed in the data.

    Returns:
    - Estimated phase (phi) in radians.
    """
    # Ensure amplitudes are a numpy array
    amplitudes = np.asarray(amplitudes)

    # Find zero-crossing indices
    zero_crossings_indices = np.where(np.diff(np.sign(amplitudes)))[0]
    
    if len(zero_crossings_indices) < zero_crossings:
        raise ValueError("Not enough zero crossings found in the data.")

    # Calculate the average period of the oscillation
    periods = np.diff(zero_crossings_indices)
    average_period = np.mean(periods)
    
    # Calculate the phase shift
    # Phase shift is based on the position of the zero crossings relative to the expected period
    expected_period = 2 * np.pi / omega
    num_cycles = average_period / expected_period
    phase_shift = (num_cycles - int(num_cycles)) * 2 * np.pi  # Phase shift in radians
    
    return phase_shift

