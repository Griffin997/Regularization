############# Libaries ###############

import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import numpy as np
import statistics
import math
import time
import itertools
import pickle
from tqdm import trange
from datetime import date

from itertools import product, zip_longest
import multiprocessing

import os

import pandas as pd

import sys

from tqdm import tqdm, trange



import multiprocess as mp
# from multiprocess import RawArray

from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
# import torch.multiprocessing as mpp

# import time as time2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
parent = os.path.dirname(os.path.abspath(''))
sys.path.append(parent)
# import config
# import noisySignalGen
# import pickle
# import bz2
# import sys
# from DF_DataLoader import initDataset
# from scipy.optimize import curve_fit
# import ray
# from ray.util.multiprocessing import Pool
import functools

############# Global Params ###############

noise_date_oi = "28Nov22"

with open('SimulationSets//standardNoise_' + noise_date_oi + '.pkl', 'rb') as handle:
    noise_mat = pickle.load(handle)
    noise_mat = np.array(noise_mat)
handle.close()

SNR_mat = [50, 100]
n_elements = 128
#Weighting term to ensure the c_i and T2_i are roughly the same magnitude
ob_weight = 100
n_noise_realizations = 5 #500

num_multistarts = 10

agg_weights = np.array([1, 1, 1/ob_weight, 1/ob_weight])

upper_bound = [2,2,100,300] #Set upper bound on parameters c1, c2, T21, T22, respectively
initial = (0.5, 0.5, 30, 150) #Set initial guesses

tdata = np.linspace(0, 635, n_elements)
lambdas = np.append(0, np.logspace(-7,1,51))

# Parameters Loop Through
c1_set = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
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


# triplets = product(c1_set, c2_set, T21_set,T22_set)
# list_tuple_store = [x for x in triplets]
# np_triplets = np.array(list_tuple_store)
# indexes_m = np.arange(0, triplets.shape[0]).reshape(-1,1)
# new_target_iterator = np.hstack((np_triplets, indexes_m))


def construct_paramList(c1_list, T21_list, T22_list):
    preList = [item for item in itertools.product(c1_list, T21_list, T22_list)]
    postList = [list(elem) for elem in preList]
    [elem.insert(1,1-elem[0]) for elem in postList]
    postList = np.array(postList)
    indexes_m = np.arange(0, postList.shape[0]).reshape(-1,1)
    new_target_iterator = np.hstack((postList, indexes_m))
    return new_target_iterator

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
    # T21_ld = np.where(T21_ld > T22_ld, T22_ld, T21_ld)
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
    all_estimates = np.zeros((noise_mat.shape[0], len(lambdas), 4))
    all_RSS = np.zeros((noise_mat.shape[0], len(lambdas)))
    
    underlying = G(tdata, *param_combo)
    all_noise = underlying + (1/iSNR)*noise_mat

    for nr in range(n_noise_realizations):    #Loop through all noise realizations
        noise_data = all_noise[nr,:]

        for iLam in range(len(lambdas)):    #Loop through all lambda values
            lam = lambdas[iLam]
            param_estimates = estimate_parameters(noise_data, lam)
            all_estimates[nr, iLam, :] = param_estimates

            estimated_model = G(tdata, *param_estimates)
            all_RSS[nr, iLam] = np.sum((estimated_model - noise_data)**2)

    estimates_df = pd.DataFrame(columns =["TrueParam", "Estimates", "RSS"])
    estimates_df["TrueParam"] = [param_combo]
    estimates_df["Estimates"] = [all_estimates]
    estimates_df["RSS"] = [all_RSS]
    return estimates_df




for iSNR in SNR_mat:    #Loop through different SNR values

    if __name__ == '__main__':
        freeze_support()




        # assert len(data.shape == 2) 
        # X = RawArray('d', data.shape[0]*data.shape[1])
        # X_np = np.frombuffer(X).reshape(data.shape)
        # np.copyto(X_np, data)
        # print(array_id)



        # result_ids = [mycurvefit_l2Regularized_2param.remote(array_id, i) for i in trange(10_000_000)]

        print("Finished Assignments...")
        # output = ray.get(result_ids) 
        # with tqdm(total = noisy_signals.shape[0]) as pbar:

        #     for (t21, t22),k in pool.imap_unordered(mycurvefit_l2Regularized_2param,j_NDs):
        #         lis.append(torch.tensor((t21,t22, torch.tensor(k))))
        #         pbar.update()
        # print(output.shape)
        # T2_ld = torch.stack([row for row in lis], dim = 0)
        # print(T2_ld.shape) #should be num_triplets X num_realizations

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
        #             # break
            pool.close() #figure out how to implement
            pool.join()
        # # assert False
        # T2_ld = np.stack([row for row in lis], axis = 0)
        print(len(lis)) #should be target_iterator.shape[0]
        df = pd.concat(lis, ignore_index= True)

        # df[df.columns] = T2_ld
        # df.set_index("Index", inplace=True, drop = True)
        # df.sort_index(inplace = True)
#             if option == 0:
#                 df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_Lambda0_SNR_900.0_{dp}Data.feather")
#             elif option ==1:
        df.to_feather(f'SimulationSets//runInfo_SNR{iSNR}_' + day + month + year + '.feather')           



def to_readable(file):
    df_read = pd.read_feather(file)
    ALL_ESTIMATES = np.stack(df["Estimates"].values) #shape: (index, noise_realization, lambda, popt)
    ALL_RES = np.stack(df["RSS"].values) #shape: (index, noise_realization, lambda, residual)
    ALL_PARAMS = np.stack(df["TrueParams"].values) #(index, 4)
    return ALL_PARAMS, ALL_ESTIMATES, ALL_RES




############## Body Code ################



runInfo = {
    "SNR_array": SNR,
    'num_noise_realizations': n_noise_realizations,
    'lambdas': lambdas,
    "c1_array": c1_set,
    "T21_array": T21_set,
    "T22_array": T22_set,
    'times': tdata        
}

paramCombos = construct_paramList(c1_set, T21_set, T22_set)
runInfo["param_combos"] = paramCombos

complete_estimates, complete_RSS = generate_all_estimates(paramCombos, SNR, noise_mat, lambdas)
runInfo["complete_estimates"] = complete_estimates
runInfo["complete_RSS"] = complete_RSS

f = open(f'SimulationSets//runInfo_' + day + month + year +'.pkl','wb')
pickle.dump(runInfo,f)
f.close()