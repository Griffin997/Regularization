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
import functools
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

add_noise = True              #True for a standard reference and False for a noise set
add_mask = False                #Add a mask to the data - this mask eliminates data below a threshold (mas_amplitude)
apply_normalizer = True        #Normalizes the data during the processing step
subsection = False              #Looks at a region a sixteenth of the full size
multistart_method = False       #Applies a multistart method for each parameter fitting instance
MB_model = False                #This model incoroporates the normalization and offset to a three parameter fit
model_selection = True         #Compares monoX and biX to be able to choose fit process
testCase = True

# The MB_model does the normalization as part of the algorithm
if MB_model: assert(not apply_normalizer)

############## Frequently Changed Parameters ###########

n_lambdas = 101
lambdas = np.append(0, np.logspace(-5,1, n_lambdas))

SNR_goal = 100

addTag = 'small2_'

#There are 8 cpus available on my personal computer
num_cpus_avail = 3

if not add_noise:
    iterations = 1
else:
    iterations = 3

############## Initializing Data ##########

file_oi = "BIC_triTest.mat"#"BIC_triTest.mat"#"NESMA_cropped_slice5.mat"
folder_oi = "BIC_tests"#"BIC_tests"#"BLSA_1742_04_MCIAD_m41"
specific_name = 'BIC_triTest'#"BIC_triTest"#'slice_oi' - this is important if the data strux has an internal name

output_folder = "ExperimentalSets"

brain_data = scipy.io.loadmat(os.getcwd() + f'\\MB_References\\{folder_oi}\\{file_oi}')
I_raw = brain_data[specific_name]

if subsection:
    I_raw_vert = 5
    I_raw_hori = 36
    I_raw_extent = 60
    I_raw = I_raw[I_raw_vert:I_raw_vert + I_raw_extent, I_raw_hori:I_raw_hori + I_raw_extent, :]

n_vert, n_hori, n_elements_brain = I_raw.shape

t_increment_brain = 11.3 #This is a measurement originally indicated by Chuan Bi in the initial email about this data
TDATA = np.linspace(t_increment_brain, (n_elements_brain)*(t_increment_brain), n_elements_brain)

#This is how we will keep track of all voxels that are called
target_iterator = np.array([item for item in itertools.product(np.arange(0,n_vert,1), np.arange(0,n_hori,1))])

############# Global Parameters ###############

# all pixels with a lower mask amplitude are considered to be free water (i.e. vesicles)
mask_amplitude = 750    #Might need to be greater

ob_weight = 100

if multistart_method:
    num_multistarts = 2
else:
    num_multistarts = 1

############# Standard Region of Interest ###############

SNR_collect = []

if testCase:
    vert1 = 0
    vert2 = 9
    hori1 = 0
    hori2 = 9
elif subsection:
    vert1 = 33
    vert2 = 43
    hori1 = 25
    hori2 = 59
else:
    vert1 = 90             #60     #108
    vert2 = 110            #125     #116
    hori1 = 70            #100      #86
    hori2 = 130            #115      #93

vBox = (vert1,vert1,vert2,vert2,vert1)
hBox = (hori1,hori2,hori2,hori1,hori1)

noiseRegion = [vert1,vert2,hori1,hori2]

####### File Naming Section #########

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

if not apply_normalizer and not MB_model:
    seriesTag = (seriesTag + "NoNorm" + "_")

if testCase:
    seriesTag = (seriesTag + "testCase" + "_")



seriesTag = (seriesTag + addTag + day + month + year)

seriesFolder = (os.getcwd() + f'/{output_folder}/{seriesTag}')
os.makedirs(seriesFolder, exist_ok = True)

############# Signal Functions ##############

def G_biX_off(t, con_1, con_2, tau_1, tau_2, offSet): 
    signal = con_1*np.exp(-t/tau_1) + con_2*np.exp(-t/tau_2) + offSet
    return signal

def G_moX_off(t, con, tau, offSet): 
    signal = con*np.exp(-t/tau) + offSet
    return signal

def G_MB(t, alpha, beta, tau_1, tau_2, offSet):
    function = alpha*(beta*np.exp(-t/tau_1) + (1-beta)*np.exp(-t/tau_2)) + offSet
    return function

def G_reg(lam, func, SA = 1):
    #SA defines the signal amplitude, defaults to 1 with assumed normalized data
    #Regularization is only applied to biexponential data
    f_name = func.__name__
    if 'biX' in f_name:
        def Gt_lam(t, con_1, con_2, tau_1, tau_2, offSet):
            param_stack = [lam*con_1/SA, lam*con_2/SA, lam*tau_1/ob_weight, lam*tau_2/ob_weight]
            return np.append(G_biX_off(t, con_1, con_2, tau_1, tau_2, offSet), param_stack)
    elif 'MB' in f_name:
        def Gt_lam(t, alpha, beta, tau1, tau2, oS):
            param_stack = [lam*alpha/SA, lam*beta, lam*tau1/ob_weight, lam*tau2/ob_weight]
            return np.append(G_MB(t, alpha, beta, tau1, tau2, oS), param_stack)
    else:
        raise Exception("Not a valid function: " + f_name)
    return Gt_lam

def G_reg_param(lam, func, popt, SA = 1):
    #SA defines the signal amplitude, defaults to 1 with assumed normalized data
    #Regularization is only applied to biexponential data
    f_name = func.__name__
    if 'biX' in f_name:
        param_stack = popt[:4]*np.array([lam/SA, lam/SA, lam/ob_weight, lam/ob_weight])
    elif 'MB' in f_name:
        param_stack = popt[:4]*np.array([lam/SA, lam, lam/ob_weight, lam/ob_weight])
    else:
        raise Exception("Not a valid function: " + f_name)
    return param_stack

############# Selecting Function ###############

if MB_model:
    model_oi = G_MB
else:
    model_oi = G_biX_off

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

    if regionEnd_std == 0:
        SNR_region = np.inf
    else:
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

def add_noise_brain_uniform(raw, SNR_desired, region, I_mask_factor, noise_seed):
    #This function was built with the intention of taking a region (e.g. the homogenous region to the right of the ventricles)
    #Add noise to make sure the final SNR is close to the desired SNR

    np.random.seed(noise_seed)

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

################## Parameter Estimation Helper Functions ###############

def get_param_p0(func, sig_init = 1, rand_opt = multistart_method):
    f_name = func.__name__
       
    if 'biX' in f_name:
        if rand_opt:
            init_p0 = [sig_init*0.2, sig_init*0.8, 20, 80, 1]
        else:
            init_p0 = [sig_init*0.2, sig_init*0.8, 20, 80, 1]
    elif 'moX' in f_name:
        if rand_opt:
            init_p0 = [sig_init, 20, 1]
        else:
            init_p0 = [sig_init, 20, 1]
    elif 'MB' in f_name:
        if rand_opt:
            init_p0 = [sig_init, 0.2, 20, 80, 1]
        else:
            init_p0 = [sig_init, 0.2, 20, 80, 1]
    else:
        raise Exception("Not a valid function: " + f_name)

    return init_p0

def get_upperBound(func, sig_init = 1):
    #These bounds were chosen to match the simulated data while also being restrictive enough
    #This provides a little extra space as the hard bounds would be [1,1,50,300]
    f_name = func.__name__
       
    if 'biX' in f_name:
        init_p0 = [0.75*sig_init, 2*sig_init, 80, 300, np.inf]
    elif 'moX' in f_name:
        init_p0 = [1.5*sig_init, 300, np.inf]
    elif 'MB' in f_name:
        init_p0 = [np.inf, 0.5, 80, 2000, np.inf]
    else:
        raise Exception("Not a valid function: " + f_name)

    return init_p0

def check_param_order(popt, func):
    #Function to automate the order of parameters if desired
    #Reshaping of array to ensure that the parameter pairs all end up in the appropriate place - ensures that T22 > T21
    
    f_name = func.__name__
    num = 0
    if 'off' in f_name:
        num = -1

    if 'MB' in f_name:
        return popt

    if (popt[-2+num] > popt[-1+num]): #We want by convention to make sure that T21 is <= T22
        for i in range(popt.shape[0]//2):
            p_hold = popt[2*i]
            popt[2*i] = popt[2*i+1]
            popt[2*i+1] = p_hold
    return popt

def calc_Radu_SNR(sig, est_curve):
    residuals = sig - est_curve
    curve_std = np.max([np.std(residuals), 10**-16])
    return sig[0]/curve_std

def calculate_RSS(data, popt, func):
    est_curve = func(TDATA, *popt)
    RSS = np.sum((est_curve - data)**2)
    
    return RSS

def calculate_reg_RSS(data, popt, func, lam):
    est_curve = func(TDATA, *popt)

    param_resid = G_reg_param(lam, func, popt, SA = data[0])

    RSS = np.sum((est_curve - data)**2)
    param_RSS = np.sum(param_resid**2)

    return RSS + param_RSS

def calculate_BIC(RSS, popt, sigma):

    BIC = 1/TDATA.shape[0] * (RSS + np.log(TDATA.shape[0]) * popt.shape[0]*(sigma)**2)

    return RSS

################## Parameter Estimation Functions ###############

def single_reg_param_est(data, lam, func):
            
    init_p = get_param_p0(func, sig_init = data[0])
    upper_bound = get_upperBound(func)
    lower_bound = np.zeros(len(upper_bound))
    parameter_tail = np.zeros(len(upper_bound)-1)
    data_tilde = np.append(data, parameter_tail) # Adds zeros to the end of the regularization array for the param estimation
    
    try:
        popt, _ = curve_fit(G_reg(lam, func, SA = data_tilde[0]), TDATA, data_tilde, bounds = (lower_bound, upper_bound), p0=init_p, max_nfev = 4000)
    except Exception as error:
        popt = [0,0,1,1,0]
        print("Error in parameter fitting: " + str(error))

    return popt

def perform_multi_estimates(data, lam, func, n_initials = num_multistarts):
    #Pick n_initials random initial conditions within the bound, and choose the one giving the lowest model-data mismatch residual

    RSS_hold = np.inf
    reg_RSS_hold = np.inf
    for i in range(n_initials):

        popt = single_reg_param_est(data, lam, func)

        reg_RSS_temp = calculate_reg_RSS(data, popt, func, lam)

        RSS_temp = calculate_RSS(data, popt, func)

        if reg_RSS_temp < reg_RSS_hold:
            reg_RSS_hold = reg_RSS_temp
            best_popt = popt
            RSS_hold = RSS_temp
        
    popt = check_param_order(best_popt, func)
 
    return popt, RSS_hold



def BIC_filter(data):

    biX_upperBounds = get_upperBound(G_biX_off)
    moX_upperBounds = get_upperBound(G_moX_off)

    biX_lowerBounds = np.zeros(len(biX_upperBounds))
    moX_lowerBounds = np.zeros(len(moX_upperBounds))

    biX_initP = get_param_p0(G_biX_off, sig_init = data[0])
    moX_initP = get_param_p0(G_moX_off, sig_init = data[0])

    G_biX_off_params, _ = curve_fit(G_biX_off, TDATA, data, bounds = (biX_lowerBounds, biX_upperBounds), p0=biX_initP, max_nfev = 4000)
    G_moX_off_params, _ = curve_fit(G_moX_off, TDATA, data, bounds = (moX_lowerBounds, moX_upperBounds), p0=moX_initP, max_nfev = 4000)

    curve_SNR = calc_Radu_SNR(data, G_biX_off(TDATA, *G_biX_off_params))

    RSS_biX = calculate_RSS(data, G_biX_off_params, G_biX_off)
    RSS_moX = calculate_RSS(data, G_moX_off_params, G_moX_off)

    BIC_G_biX = calculate_BIC(RSS_biX, G_biX_off_params, data[0]/curve_SNR)
    BIC_G_moX = calculate_BIC(RSS_moX, G_moX_off_params, data[0]/curve_SNR)

    return BIC_G_moX < BIC_G_biX, G_moX_off_params, BIC_G_moX


def main_estimator(i_voxel, full_brain_data, func):
    i_vert, i_hori = target_iterator[i_voxel]
    data = full_brain_data[i_vert, i_hori, :]

    # elem_lis = []
    feature_df = pd.DataFrame(columns = ["Data", "Indices", "Type", "Params", "RSS"])
    feature_df["Data"] = [data]
    feature_df["Indices"] = [[i_vert, i_hori]]

    #Catch all to remove any background pixels from evaluation - stops the calculations early
    if data[0] == 0:
        feature_df["Type"] = ["background"]
        # elem_lis.append(feature_df)
        # return pd.concat(elem_lis, ignore_index= True)
        return feature_df
    
    ##### Flag for model selection
    if model_selection:
        BIC_boolean, G_moX_off_params, RSS_moX = BIC_filter(data)
    else:
        BIC_boolean = False

    #BIC Model Selection
    if BIC_boolean:
        feature_df["Type"] = ["moX"]
        # elem_lis.append(feature_df)

        # lam_df = pd.DataFrame(columns = ["Params", "RSS"])
        feature_df["Params"] = [G_moX_off_params]
        feature_df["RSS"] = [RSS_moX]
        # elem_lis.append(lam_df)
    else:
        feature_df["Type"] = ["biX"]
        # elem_lis.append(feature_df)

        RSS_list = []
        param_estimates_list = []
        for lam in lambdas:    #Loop through all lambda values
            param_estimates, RSS_estimate = perform_multi_estimates(data, lam, func)

            # lam_df = pd.DataFrame(columns = ["Params", "RSS"])
            
            RSS_list.append(RSS_estimate)
            param_estimates_list.append(param_estimates)

        feature_df["Params"] = [np.array(param_estimates_list)]
        feature_df["RSS"] = [RSS_list]

    # return pd.concat(elem_lis, ignore_index= True)
    return feature_df


#### This ensures that the same mask is applied throughout

if add_mask:
    I_masked = mask_data(I_raw, mask_amplitude)
else:
    I_masked = I_raw
I_mask_factor = (I_masked!=0)

#### Looping through Iterations of the brain - applying parallel processing to improve the speed
if __name__ == '__main__':
    freeze_support()

    for iter in range(iterations):    #Build {iterations} number of noisey brain realizations

        np.random.seed(iter)

        if add_noise:
            I_noised = add_noise_brain_uniform(I_masked, SNR_goal, noiseRegion, I_mask_factor, iter)[0]
        else:
            I_noised = I_masked

        if apply_normalizer:
            noise_iteration = normalize_brain(I_noised)
        else:
            noise_iteration = I_noised

        SNR_collect.append(calculate_brain_SNR(noise_iteration, noiseRegion))


        print("Finished Assignments...")

        ##### Set number of CPUs that we take advantage of
        
        if num_cpus_avail >= 10:
            print("Using Super Computer")

        print(f"Building {iter+1} Dataset of {iterations}...")
        lis = []

        with mp.Pool(processes = num_cpus_avail) as pool:

            with tqdm(total=target_iterator.shape[0]) as pbar:
                for estimates_dataframe in pool.imap_unordered(functools.partial(main_estimator, full_brain_data = noise_iteration, func = model_oi), range(target_iterator.shape[0])):
                # for estimates_dataframe in pool.imap_unordered(lambda hold_iter: main_estimator(hold_iter, noise_iteration, model_oi), range(target_iterator.shape[0])):

                    lis.append(estimates_dataframe)

                    pbar.update()

            pool.close()
            pool.join()
        

        print(f"Completed {len(lis)} of {target_iterator.shape[0]} voxels") #should be target_iterator.shape[0]
        df = pd.concat(lis, ignore_index= True)

        df.to_pickle(seriesFolder + f'/brainData_' + seriesTag + f'_iteration_{iter}.pkl')     
    

############## Save General Code ################

hprParams = {
    "SNR_goal": SNR_goal,
    'n_noise_realizations': iterations,
    'lambdas': lambdas,
    "data_file": folder_oi,
    "data_slice": file_oi,
    'tdata': TDATA,
    'ob_weight': ob_weight,
    'num_multistarts': num_multistarts,
    'model_oi': model_oi,
    'upper_bound': get_upperBound(model_oi),
    'mask_amp': mask_amplitude,
    'masked_region': I_mask_factor,
    'n_horizontal': n_hori,
    'n_verticle': n_vert,
    'options': [add_noise, add_mask, apply_normalizer, 
                subsection, MB_model, model_selection,
                multistart_method, testCase],
    'SNR_array': SNR_collect
}

f = open(seriesFolder + '/hprParameter_info_' + day + month + year +'.pkl','wb')
pickle.dump(hprParams,f)
f.close()