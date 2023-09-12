import random
import numpy as np
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


#set seed
np.random.seed(10)
random.seed(10)

# Plot empircal distribution of markov chain
n1,n2 = 100,100

#Train and find MCMC proposal distribution
# Generate synthetic data
samples1 = np.random.normal(5.75, 1.5, 1000)
samples2 = np.random.normal(9.2, 0.5, 1000)
combined_samples = np.concatenate((samples1, samples2)).reshape(-1, 1)

# Fit a Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2)
gmm.fit(combined_samples)
    


def target_function(x):
    return min(15000,max(0,50*(x-8)**5))

def TARIS_GMM_PROPOSAL(T,N):
    y = 5
    duration = np.arange(T)
    first_ratio = 5
    ratio_estimate_per_iteration = np.zeros(T)
    ratio_estimate_averages = np.zeros(T-5)
    ratio_estimate_per_iteration[0] = first_ratio
    first_x = np.zeros(1)
    for t in duration:
        print("starting t at ", datetime.now())
        x = [np.array(7.5)] #[0.5 * (np.mean(q_1_samples) + np.mean(q_2_samples))]
        if t == 0:
            ratio_estimate_per_iteration_now = ratio_estimate_per_iteration[0]
            x = [np.array(7.5)]
        else:
            ratio_estimate_per_iteration_now = ratio_estimate_per_iteration[t-1]
            x = [first_x]
        length = np.arange(N)
        for i in length:
            if i%10000 == 0:
                print(i) 
            #propose new X'
            x_old = x[-1]
            x_new = gmm.sample(1)[0][0]
            #calculate acceptance probability
            nom_accept = abs(target_function(x_new)-ratio_estimate_per_iteration_now) * sp.stats.gamma.pdf(x_new, a=5, loc=0, scale=4) * sp.stats.norm.pdf(y, loc=x_new, scale=1) * np.exp(gmm.score_samples(np.array(x_old).reshape(-1, 1)))
            denom_accept = abs(target_function(x_old)-ratio_estimate_per_iteration_now) * sp.stats.gamma.pdf(x_old, a=5, loc=0, scale=4) * sp.stats.norm.pdf(y, loc=x_old, scale=1) * np.exp(gmm.score_samples(np.array(x_new).reshape(-1, 1)))
            alpha = min(1,nom_accept/denom_accept)
            u = np.random.uniform(low=0.0, high=1.0, size=1)
            if u <= alpha:
                x.append(x_new)
                
            elif u > alpha:
                x.append(x_old)
        
        
        x = x[1000:]
        first_x = x[-1]          #save last x for next iteration
        f_x_here = [target_function(entry[0]) for entry in x]
        f_x_here = np.array(f_x_here)
        nom_rat = np.sum(f_x_here/abs(f_x_here - ratio_estimate_per_iteration_now))
        denom_rat = np.sum(1/abs(f_x_here - ratio_estimate_per_iteration_now))
        r_current = nom_rat/denom_rat
        ratio_estimate_per_iteration[t] = r_current
        if t < 5:
            pass
        else:
            r_estimates_until_now = np.concatenate((ratio_estimate_per_iteration[5:t], np.array([r_current])))
            ratio_estimate_averages[t-5] = np.mean(r_estimates_until_now)
        print('iteration',t+1,'is done')
    return {'ratio_estimate_per_iteration': ratio_estimate_per_iteration, 'ratio_estimate_averages': ratio_estimate_averages}


def calculate_groundtruth():
    # Parameters for p(x) = Gamma(x; shape=5, scale=4)
    shape_param = 5
    scale_param = 4

    # Parameters for p(y|x) = Normal(y; x, 1)
    observed_y = 5

    # Define function f(x)
    def f(x):
        return min(15000, max(0, 50 * (x - 8)**5))

    # Posterior density function
    def posterior_density(x):
        return gamma.pdf(x, a=shape_param, scale=scale_param) * norm.pdf(observed_y, loc=x, scale=1) / quad(lambda x: gamma.pdf(x, a=shape_param, scale=scale_param) * norm.pdf(observed_y, loc=x, scale=1), -np.inf, np.inf)[0]

    # Perform numerical integration using scipy.integrate.quad
    expected_value, _ = quad(lambda x: f(x) * posterior_density(x), -np.inf, np.inf)
    return expected_value

def calculate_TARIS_ReMSE(result_100, result_1k, result_10k):
    ground_truth = calculate_groundtruth()
    
    #extract results
    r_average_100 = result_100['ratio_estimate_averages']

    r_average_1k = result_1k['ratio_estimate_averages']

    r_average_10k = result_10k['ratio_estimate_averages']
    
    #calculate ReMSE
    
    ReMSE_10k = (r_average_10k - ground_truth)**2/ground_truth**2
    ReMSE_1k = (r_average_1k - ground_truth)**2/ground_truth**2
    ReMSE_100 = (r_average_100 - ground_truth)**2/ground_truth**2

    total_samples_1k = np.arange(1000, 1000*(len(r_average_1k)+1), 1000)
    total_samples_10k = np.arange(10000, 10000*(len(r_average_10k)+1), 10000)
    total_samples_100 = np.arange(100, 100*(len(r_average_100)+1), 100)
    return ReMSE_10k, ReMSE_1k, ReMSE_100, total_samples_1k, total_samples_10k, total_samples_100


def compute_TABI(n1,n2):
    np.random.seed(1)
    random.seed(1)
        
    ground_truth = calculate_groundtruth()
    q_1_samples = nct.rvs(df=10, nc=0, loc=9.3,scale=0.5, size=n1)
    q_2_samples = np.random.normal(loc=5.4, scale=0.98, size=n2)

    joint_q1 = sp.stats.gamma.pdf(q_1_samples, a=5, loc=0, scale=4) * sp.stats.norm.pdf(5, loc=q_1_samples, scale=1)
    joint_q2 = sp.stats.gamma.pdf(q_2_samples, a=5, loc=0, scale=4) * sp.stats.norm.pdf(5, loc=q_2_samples, scale=1)

    def target_function(x):
        return min(15000,max(0,50*(x-8)**5))
    
    target_function_q1 = np.array(list(map(target_function, q_1_samples)))

    q_1_density = nct.pdf(q_1_samples,df=10, nc=0, loc=9.3,scale=0.5)
    q_2_density = sp.stats.norm.pdf(q_2_samples, loc=5.4, scale=0.98)

    nominator = 1/n1 * np.sum(target_function_q1 * joint_q1/q_1_density)
    denominator = 1/n2 * np.sum(joint_q2/q_2_density)
    ratio = nominator/denominator
    ReMSE = (ratio - ground_truth)**2/ground_truth**2

    return {'ratio': ratio, 'ReMSE': ReMSE}