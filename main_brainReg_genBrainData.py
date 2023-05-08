############# Libaries ###############

import h5py
import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
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

############# Data Set Options & Hyperparameters ############

add_noise = False          #Add noise to the data beyond what is there naturally
add_mask = True             #Add a mask to the data - this mask eliminates data below a threshold (mas_amplitude)
apply_normalizer = True     #Normalizes the data during the processing step
estimate_offset = True      #Adds an offset to the signal that is estimated
subsection = False           #Looks at a region a sixteenth of the full size
multistart_method = False    #Applies a multistart method for each parameter fitting instance
MB_model = False           #This model incoroporates the normalization and offset to a three parameter fit

# The MB_model does the normalization as part of the algorithm
if MB_model: assert(not apply_normalizer and not estimate_offset)

############## Frequently Changed Parameters ###########

n_lambdas = 101

SNR_goal = 100

if add_noise:
    iterations = 20
else:
    iterations = 1

############## Initializing Data ##########

brain_data = scipy.io.loadmat(os.getcwd() + '/MB_References/BLSA_1742_04_MCIAD_m41/NESMA_slice5.mat')
I_raw = brain_data['slice_oi']

if subsection:
    I_raw_x = I_raw.shape[0]//4
    I_raw_y = I_raw.shape[1]//4
    I_raw = I_raw[I_raw_x:2*I_raw_x, I_raw_y:2*I_raw_y, :]

n_vert, n_hori, n_elements_brain = I_raw.shape

t_increment_brain = 11.3 #This is a measurement originally indicated by Chuan Bi in the initial email about this data
tdata = np.linspace(t_increment_brain, (n_elements_brain)*(t_increment_brain), n_elements_brain)

#This is how we will keep track of all voxels that are called
target_iterator = np.array([item for item in itertools.product(np.arange(0,n_vert,1), np.arange(0,n_hori,1))])

#NESMA Filter parameters
txy = 3
# tz = 5  #unused in the 2D scans in this code
thresh = 5
# all pixels with a lower mask amplitude are considered to be free water (i.e. vesicles)
mask_amplitude = 750

############# Global Params ###############

#These bounds were chosen to match the simulated data while also being restrictive enough
#This provides a little extra space as the hard bounds would be [1,1,50,300]
if MB_model:
    upper_bound = [np.inf, 0.5, 60, 2000] #c1 used to be 0.5 - changed to 1 for complete range
elif not apply_normalizer:
    upper_bound = [0.5,1.2,60,300]
else:
    upper_bound = [0.5,1.2,60,300]

if estimate_offset or MB_model:
    upper_bound.append(np.inf)

lambdas = np.append(0, np.logspace(-5,1, n_lambdas))

ob_weight = 100
if MB_model:
    agg_weights = np.array([1/ob_weight, 1, 1/ob_weight, 1/ob_weight])
if not apply_normalizer:
    agg_weights = np.array([1,1,1,1])
else:
    agg_weights = np.array([1, 1, 1/ob_weight, 1/ob_weight])

if multistart_method:
    num_nultistarts = 10
    ms_upper_bound = [1,60,300] 
else:
    num_multistarts = 1
    ms_upper_bound = [0] 

 

SNR_collect = np.zeros(iterations)

if subsection:
    vert1 = 37
    vert2 = 47
    hori1 = 25
    hori2 = 70
else:
    vert1 = 165             #60     #108
    vert2 = 180            #125     #116
    hori1 = 120            #100      #86
    hori2 = 180            #115      #93

vBox = (vert1,vert1,vert2,vert2,vert1)
hBox = (hori1,hori2,hori2,hori1,hori1)

noiseRegion = [vert1,vert2,hori1,hori2]

# Important for Naming
date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

seriesTag = ""
if add_noise:
    seriesTag = (seriesTag + f"SNR_{SNR_goal}_")
else:
    seriesTag = (seriesTag + f"NoNoise_")

if subsection:
    seriesTag = (seriesTag + f"subsection_")

if upper_bound[0]==0.5:
    seriesTag = (seriesTag + "halfC" + "_")

if not estimate_offset and not MB_model:
    seriesTag = (seriesTag + "noOffSet" + "_")

if MB_model:
    seriesTag = (seriesTag + "MBmod" + "_")

if not apply_normalizer and not MB_model:
    seriesTag = (seriesTag + "NoNorm" + "_")

# seriesTag = (seriesTag + "2000T22" + "_")

seriesTag = (seriesTag + day + month + year)

seriesFolder = (os.getcwd() + '/ExperimentalSets/' + seriesTag)
os.makedirs(seriesFolder, exist_ok = True)

############# Signal Functions ##############

def G(t, con_1, con_2, tau_1, tau_2): 
    function = con_1*np.exp(-t/tau_1) + con_2*np.exp(-t/tau_2)
    return function

def G_off(t, con_1, con_2, tau_1, tau_2, offSet): 
    function = con_1*np.exp(-t/tau_1) + con_2*np.exp(-t/tau_2) + offSet
    return function

def G_MB(t, alpha, beta, tau_1, tau_2, offSet):
    function = alpha*(beta*np.exp(-t/tau_1) + (1-beta)*np.exp(-t/tau_2)) + offSet
    return function

def G_tilde(lam, SA = 1, offSet = estimate_offset, opt_MB = MB_model):
    #SA defines the signal amplitude, defaults to 1 for simulated data
    if offSet:
        def Gt_lam(t, con1, con2, tau1, tau2, oS):
            return np.append(G_off(t, con1, con2, tau1, tau2, oS), [lam*con1/SA, lam*con2/SA, lam*tau1/ob_weight, lam*tau2/ob_weight])
    elif opt_MB:
        def Gt_lam(t, alpha, beta, tau1, tau2, oS):
            return np.append(G_MB(t, alpha, beta, tau1, tau2, oS), [lam*alpha/ob_weight, lam*beta/SA, lam*tau1/ob_weight, lam*tau2/ob_weight])
    else:
        def Gt_lam(t, con1, con2, tau1, tau2):
            return np.append(G(t, con1, con2, tau1, tau2), [lam*con1/SA, lam*con2/SA, lam*tau1/ob_weight, lam*tau2/ob_weight])
    return Gt_lam

############# Data Processing Functions ##############

def mask_data(raw, mask_amplitude):
    #Sets every decay curve in the data set where the amplitude is less than a threshold value to zero
    I_masked = np.copy(raw)
    I_masked[I_masked[:,:,0]<mask_amplitude] = 0
    return I_masked

def calculate_brain_SNR(raw, region):
    #calculates the SNR of the brain using a homogenous region fed into the 
    v1,v2,h1,h2 = region

    rawZone = raw[v1:v2,h1:h2,:]

    regionZero = rawZone[:, :, 0]
    regionZero_mean = np.mean(regionZero)

    regionEnd = rawZone[:, :, -3:] #last three points across the entire sampled region
    regionEnd_std = np.std(regionEnd)
    regionEnd_absMean = np.mean(np.abs(regionEnd))

    SNR_region = (regionZero_mean - regionEnd_absMean)/regionEnd_std

    return SNR_region

def normalize_brain(I_data):
    n_vert, n_hori, n_elem = I_data.shape
    I_normalized = np.zeros(I_data.shape)
    for i_vert in range(n_hori):
        for i_hori in range(n_vert):
            data = I_data[i_vert,i_hori,:]
            if data[0]>0:
                data_normalized = data/(data[0])
            else:
                data_normalized = np.zeros(n_elements_brain)
            I_normalized[i_vert,i_hori,:] = data_normalized
    return I_normalized

def add_noise_brain_uniform(raw, SNR_desired, region, I_mask_factor):
    #This function was built with the intention of taking a region (e.g. the homogenous region to the right of the ventricles)
    #Add noise to make sure the final SNR is close to the desired SNR

    v1,v2,h1,h2 = region

    rawZone = raw[v1:v2,h1:h2,:]

    regionZero = rawZone[:, :, 0]
    sigRef = np.mean(regionZero)

    regionEnd = rawZone[:, :, -3:]
    initSD = np.std(regionEnd)

    addSD = (sigRef**2/SNR_desired**2 - initSD**2)**(1/2)

    noiseMat = np.random.normal(0,addSD,raw.shape)
    I_noised = raw + noiseMat*I_mask_factor

    return I_noised, addSD

################## Parameter Estimation Functions ###############

def generate_p0(ms_ub = ms_upper_bound, offSet = estimate_offset, ms_opt = multistart_method, sig_init = 1):
    
    
    if MB_model:
        init_params = (sig_init, 0.2, 20, 80, 1)
    elif not ms_opt:
        init_params = (sig_init*0.2, sig_init*0.8, 20, 80, 1)
    else:
        three_params = np.random.uniform(0,1,3)*ms_ub
        init_params = (sig_init*three_params[0], sig_init*(1-three_params[0]), three_params[1], three_params[2], 1)

    if not offSet and not MB_model:
        init_params = init_params[0:4] #cleave the 5th parameter

    return init_params

def check_param_order(popt):
    #Function to automate the order of parameters if desired
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    if (popt[2] > popt[3]): #We want by convention to make sure that T21 is <= T22
        p_hold = popt[0]
        popt[0] = popt[1]
        popt[1] = p_hold

        p_hold = popt[2]
        popt[2] = popt[3]
        popt[3] = p_hold
    return popt

def estimate_parameters(data, lam, n_initials = num_multistarts):
    #Pick n_initials random initial conditions within the bound, and choose the one giving the lowest model-data mismatch residual

    parameter_tail = [0,0,0,0]
    data_tilde = np.append(data, parameter_tail) # Adds zeros to the end of the regularization array for the param estimation
    
    RSS_hold = np.inf
    obj_hold = np.inf
    for i in range(n_initials):

        np.random.seed(i) #Only has an effect on the multistart strategy
        if MB_model or not apply_normalizer:
            init_params = generate_p0(sig_init = data_tilde[0])
        else:
            init_params = generate_p0()

        temp_upper_bound = np.copy(upper_bound)
        if not apply_normalizer and not MB_model:
            temp_upper_bound[0] = data_tilde[0]*upper_bound[0]
            temp_upper_bound[1] = data_tilde[0]*upper_bound[1]
        
        try:
            if estimate_offset or MB_model:
                lower_bound = [0,0,0,0,0]
            else:
                lower_bound = [0,0,0,0]
            popt, _ = curve_fit(G_tilde(lam), tdata, data_tilde, bounds = (lower_bound, temp_upper_bound), p0=init_params, max_nfev = 4000)
        except Exception as error:
            if estimate_offset or MB_model:
                popt = [0,0,1,1,0]
            else:
                popt = [0,0,1,1]
            print("Error in parameter fitting: " + str(error))

        if estimate_offset:
            est_curve = G_off(tdata,*popt)
        elif MB_model:
            est_curve = G_MB(tdata,*popt)
        else:
            est_curve = G(tdata,*popt)
        RSS_temp = np.sum((est_curve - data)**2)

        obj_pTemp = lam*agg_weights*popt[0:4]
        obj_temp = RSS_temp + np.linalg.norm(obj_pTemp)

        if obj_temp < obj_hold:
            obj_hold = obj_temp
            best_popt = popt
            RSS_hold = RSS_temp
        
    if not MB_model:
        popt = check_param_order(best_popt)
    else:
        popt = best_popt
 
    return popt, RSS_hold

def generate_all_estimates(i_voxel, brain_data_3D):
    #Generates a comprehensive matrix of all parameter estimates for all param combinations, 
    #noise realizations, SNR values, and lambdas of interest
    i_vert, i_hori = target_iterator[i_voxel]
    noise_data = brain_data_3D[i_vert, i_hori, :]
    e_lis = []

    for iLam in range(len(lambdas)):    #Loop through all lambda values
        e_df = pd.DataFrame(columns = ["Data", "Indices", "Estimates", "RSS"])
        lam = lambdas[iLam]

        if np.all(noise_data == 0):
            if estimate_offset or MB_model:
                param_estimates = [0,0,1,1,0]
            else:
                param_estimates = [0,0,1,1]
            RSS_estimate = np.inf
        else:
            param_estimates, RSS_estimate = estimate_parameters(noise_data, lam)
        
        assert(noise_data.shape[0] == n_elements_brain)
        e_df["Data"] = [noise_data]
        e_df["Indices"] = [[i_vert, i_hori]]
        e_df["Estimates"] = [param_estimates]
        e_df["RSS"] = [RSS_estimate]
        e_lis.append(e_df)
    
    return pd.concat(e_lis, ignore_index= True)

#### This ensures that the same mask is applied throughout

if add_mask:
    I_masked = mask_data(I_raw, mask_amplitude)
    I_mask_factor = (I_masked!=0)
else:
    I_masked = I_raw

#### Looping through Iterations of the brain - applying parallel processing to improve the speed

for iter in range(iterations):    #Build {iterations} number of noisey brain realizations

    np.random.seed(iter)

    if add_noise:
        I_noised = add_noise_brain_uniform(I_masked, SNR_goal, noiseRegion, I_mask_factor)[0]
    else:
        I_noised = I_masked

    if apply_normalizer:
        noise_iteration = normalize_brain(I_noised)
    else:
        noise_iteration = I_noised

    SNR_collect[iter] = calculate_brain_SNR(noise_iteration, noiseRegion)

    if __name__ == '__main__':
        freeze_support()

        print("Finished Assignments...")

        num_cpus_avail = 80
        print("Using Super Computer")

        print(f"Building {iter+1} Dataset of {iterations}...")
        lis = []

        with mp.Pool(processes = num_cpus_avail) as pool:

            with tqdm(total=target_iterator.shape[0]) as pbar:
                for estimates_dataframe in pool.imap_unordered(lambda hold_label: generate_all_estimates(hold_label, noise_iteration), range(target_iterator.shape[0])):

                    lis.append(estimates_dataframe)

                    pbar.update()

            pool.close()
            pool.join()

        print(f"Completed {len(lis)} of {target_iterator.shape[0]} voxels") #should be target_iterator.shape[0]
        df = pd.concat(lis, ignore_index= True)

        df.to_feather(seriesFolder + f'/brainData_' + seriesTag + f'_iteration_{iter}.feather')           

############## Save General Code ################

hprParams = {
    "SNR_goal": SNR_goal,
    'n_noise_realizations': iterations,
    'lambdas': lambdas,
    "data_file": "BLSA_1742_04_MCIAD_m41",
    "data_slice": "rS_slice5",
    'tdata': tdata,
    'ob_weight': ob_weight,
    'agg_weights': agg_weights,
    'num_multistarts': num_multistarts,
    'upper_bound': upper_bound,
    'mask_amp': mask_amplitude,
    'ms_upBound': ms_upper_bound,
    'n_horizontal': n_hori,
    'n_verticle': n_vert,
    'options': [add_noise, add_mask, apply_normalizer, 
                estimate_offset, subsection, multistart_method, MB_model],
    'SNR_array': SNR_collect
}

f = open(seriesFolder + '/hprParameter_info_' + day + month + year +'.pkl','wb')
pickle.dump(hprParams,f)
f.close()