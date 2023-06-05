############# Libaries ###############

import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import statistics
import math
import time
import itertools
from itertools import product, zip_longest
import pickle
from tqdm import tqdm, trange
from datetime import date


import multiprocess as mp
from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
import functools

############# Global Params ###############

noise_date_oi = "28Nov22" #This is taken as the noise to keep everything standard

with open('SimulationSets//standardNoise_' + noise_date_oi + '.pkl', 'rb') as handle:
    noise_mat = pickle.load(handle)
    noise_mat = np.array(noise_mat)
handle.close()

SNR_mat = [200]
n_elements = 128
#Weighting term to ensure the c_i and T2_i are roughly the same magnitude
ob_weight = 100
n_noise_realizations = 500 #500

num_multistarts = 10

agg_weights = np.array([1, 1, 1/ob_weight, 1/ob_weight])

upper_bound = [2,2,100,300] #Set upper bound on parameters c1, c2, T21, T22, respectively
# initial = (0.5, 0.5, 30, 150) #Set initial guesses

tdata = np.linspace(0, 635, n_elements)
lambdas = np.append(0, np.logspace(-7,1,51))

# Parameters Loop Through
c1_set = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
c2_set = 1-np.array(c1_set)
T21_set = [10,20,30,40,50]
T22_set = [70,90,110,130,150]

param_name_list = ['c1','c2','T21','T22']

# Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')


############# Functions ##############

def G(t, con_1, con_2, tau_1, tau_2): 
    function = con_1*np.exp(-t/tau_1) + con_2*np.exp(-t/tau_2)
    return function

def G_tilde(lam, SA = 1):
    #SA defines the signal amplitude, defaults to 1 for simulated data
    def Gt_lam(t, con1, con2, tau1, tau2):
        return np.append(G(t, con1, con2, tau1, tau2), [lam*con1/SA, lam*con2/SA, lam*tau1/ob_weight, lam*tau2/ob_weight])
    return Gt_lam

def construct_paramList(c1_list, T21_list, T22_list):
    preList = [item for item in itertools.product(c1_list, T21_list, T22_list)]
    postList = [list(elem) for elem in preList]
    [elem.insert(1,1-elem[0]) for elem in postList]
    postList = np.array(postList)
    return postList

### Needed to process ASAP
target_iterator = construct_paramList(c1_set, T21_set, T22_set)

def check_param_order(popt):
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    if (popt[-1] < popt[-2]): #We want by convention to make sure that T21 is <= T22
        for pi in range(np.size(popt)//2):
            p_hold = popt[2*pi]
            popt[2*pi] = popt[2*pi+1]
            popt[2*pi+1] = p_hold
    return popt

def estimate_parameters(data, lam, n_initials = num_multistarts):
    #Pick n_initials random initial conditions within the bound, and choose the one giving the lowest model-data mismatch residual
    random_residuals = np.empty(n_initials)
    estimates = np.zeros((n_initials,4))
    data_start = np.abs(data[0])
    data_tilde = np.append(data, [0,0,0,0]) # Adds zeros to the end of the regularization array for the param estimation
    
    for i in range(n_initials):
        np.random.seed(i)
        ic1 = np.random.uniform(0,1)
        ic2 = 1-ic1
        ic1 = ic1*data_start
        ic2 = ic2*data_start
        iT21 = np.random.uniform(0,upper_bound[-2])
        iT22 = np.random.uniform(iT21,upper_bound[-1])
        p0 = [ic1,ic2,iT21,iT22]
        up_bnd = upper_bound*np.array([data_start, data_start, 1, 1])
        assert(np.size(up_bnd) == np.size(p0))
        
        try:
            popt, _ = curve_fit(G_tilde(lam), tdata, data_tilde, bounds = (0, up_bnd), p0=p0, max_nfev = 4000)
        except:
            popt = [0,0,1,1]
            print("Max feval reached")
        
        c1_ret, c2_ret, T21_ld, T22_ld = popt
        
        # Enforces T21 <= T22
        if T21_ld.size == 1:
            T21_ld = T21_ld.item()
            T22_ld = T22_ld.item()
            c1_ret = c1_ret.item()
            c2_ret = c2_ret.item()

        
        if T21_ld > T22_ld:
            T21_ld_new = T22_ld
            T22_ld = T21_ld
            T21_ld = T21_ld_new
            c1_ret_new = c1_ret
            c1_ret = c2_ret
            c2_ret = c1_ret_new

            assert (T21_ld != T22_ld)

        # popt = check_param_order(popt) #Require T22>T21
        estimates[i] = np.array([c1_ret, c2_ret, T21_ld, T22_ld])
        estimated_model = G(tdata, *popt)
        residual = np.sum((estimated_model - data)**2)
        random_residuals[i] = residual
    min_residual_idx = np.argmin(random_residuals)
    min_residual_estimates = estimates[min_residual_idx]
 
    return min_residual_estimates

def generate_all_estimates(i_param_combo):
    #Generates a comprehensive matrix of all parameter estimates for all param combinations, 
    #noise realizations, SNR values, and lambdas of interest
    param_combo = target_iterator[i_param_combo]
    e_lis = []
    underlying = G(tdata, *param_combo)
    all_noise = underlying + (1/iSNR)*noise_mat

    for nr in range(n_noise_realizations):    #Loop through all noise realizations
        noise_data = all_noise[nr,:]

        for iLam in range(len(lambdas)):    #Loop through all lambda values
            e_df = pd.DataFrame(columns = ["TrueParams", "Estimates", "RSS"])
            lam = lambdas[iLam]
            param_estimates = estimate_parameters(noise_data, lam)

            estimated_model = G(tdata, *param_estimates)
            one_rss = np.sum((estimated_model - noise_data)**2)
            e_df["Estimates"] = [param_estimates]
            e_df["RSS"] = [one_rss]
            e_df["TrueParams"] = [param_combo]
            e_lis.append(e_df)
    return pd.concat(e_lis, ignore_index= True)

for iSNR in SNR_mat:    #Loop through different SNR values

    if __name__ == '__main__':
        freeze_support()

        print("Finished Assignments...")

        num_cpus_avail = 80
        print("Using Super Computer")

        print(f"Building {iSNR} Dataset...")
        lis = []

        with mp.Pool(processes = num_cpus_avail) as pool:

            with tqdm(total=target_iterator.shape[0]) as pbar:
                for estimates_dataframe in pool.imap_unordered(generate_all_estimates, range(target_iterator.shape[0])):
                    # if k == 0:
                        # print("Starting...")

                    lis.append(estimates_dataframe)

                    pbar.update()

            pool.close() #figure out how to implement
            pool.join()

        print(len(lis)) #should be target_iterator.shape[0]
        df = pd.concat(lis, ignore_index= True)

        df.to_feather(f'SimulationSets//runInfo_SNR_{iSNR}_' + day + month + year + '.feather')           

############## Save General Code Code ################

hprParams = {
    "SNR_mat": SNR_mat,
    'n_noise_realizations': n_noise_realizations,
    'lambdas': lambdas,
    "c1_set": c1_set,
    "T21_set": T21_set,
    "T22_set": T22_set,
    "noise_date_oi": noise_date_oi,
    'tdata': tdata,
    'ob_weight': ob_weight,
    'num_multistarts': num_multistarts,
    'upper_bound': upper_bound
}

f = open(f'SimulationSets//hprParameter_info_' + day + month + year +'.pkl','wb')
pickle.dump(hprParams,f)
f.close()