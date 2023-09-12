import numpy as np
import random
import scipy as sp


import numpy as np
import scipy as sp
from scipy.stats import nct
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from datetime import datetime
from scipy.stats import gamma, norm
from scipy.integrate import quad


### compute ground truth for TARIS
def compute_ground_truth_TARIS(data):
    ground_truth_array = np.zeros(len(data['ground_truths']))
    for j in range(len(data['ground_truths'])):
        def f(x):
            return min(15000, max(0, 50 * (x - 4)**5))
        
        y = data['y'][j]
        mu = (5 + y) / 2
        sigma = 1 / np.sqrt(2)

        expected_value, _ = quad(lambda x: f(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf)
        ground_truth_array[j] = expected_value
    
    return ground_truth_array


### compute TARIS for different n sizes

def compute_TARIS_for_n(data, n, ground_truth_array):
    ReMSE_TARIS_array = np.zeros(len(data['ground_truths']))
    TARIS_array = np.zeros(len(data['ground_truths']))

    for j in range(len(data['ground_truths'])):
        w_q_ratio = np.exp(data['log_w_q_ratio'][j][:n])
        f_x_array = data['f_x_samples_q_ratio'][j][:n]
        
        # get f_x as array
        f_x_array = np.zeros(n)
        
        for i in range(n):
            f_x_array[i] = data['f_x_samples_q_ratio'][j][i]

        for i in range(n):
            f_x_array[i] = f_x_array[i]**2


        # Calculate estimator
        TARIS_array[j] = np.sum(f_x_array * w_q_ratio) / np.sum(w_q_ratio)

        # Calculate ReMSE
        ReMSE_TARIS_array[j] = (TARIS_array[j] - ground_truth_array[j])**2 / (ground_truth_array[j]**2)

    return ReMSE_TARIS_array, TARIS_array

### compute ground truth for TABI

def compute_ground_truth_TABI(data_1):
    ground_truth_array = np.zeros(len(data_1['ground_truths']))
    for j in range(len(data_1['ground_truths'])):
        def f(x):
            return min(15000, max(0, 50 * (x - 4)**5))
        
        y = data_1['y'][j]
        mu = (5 + y) / 2
        sigma = 1 / np.sqrt(2)

        expected_value, _ = quad(lambda x: f(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf)
        ground_truth_array[j] = expected_value
    
    return ground_truth_array

# compute TABI for different n sizes

def compute_TABI_for_n(data_1, data_2, n, ground_truth_array):
    ReMSE_TABI_array = np.zeros(len(data_1['ground_truths']))
    TABI_array = np.zeros(len(data_1['ground_truths']))

    for j in range(len(data_1['ground_truths'])):
        w_q1 = np.exp(data_1['log_w_q1'][j][:n])
        w_q2 = np.exp(data_2['log_w_q2'][j][:n])
        f_x_array = data_1['f_x_samples_q1'][j][:n]
        
        # get f_x as array
        f_x_array = np.zeros(n)

        for i in range(n):
            f_x_array[i] = data_1['f_x_samples_q1'][j][i]

        # Calculate estimator
        TABI_array[j] = np.sum(f_x_array * w_q1) / np.sum(w_q2)

        # Calculate ReMSE
        ReMSE_TABI_array[j] = (TABI_array[j] - ground_truth_array[j])**2 / (ground_truth_array[j]**2)

    return ReMSE_TABI_array, TABI_array