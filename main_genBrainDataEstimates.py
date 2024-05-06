############# Libaries ###############

import h5py
import scipy
import scipy.io
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import special
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

add_noise = False              #True for a standard reference and False for a noise set
apply_normalizer = True        #Normalizes the data during the processing step
multistart_method = False       #Applies a multistart method for each parameter fitting instance
model_selection = False         #Compares monoX and biX to be able to choose fit process

############## Frequently Changed Parameters ###########

n_lambdas = 101
lambdas = np.append(0, np.logspace(-5,1, n_lambdas))

addTag = ''

date_of_data = '25Apr24'
SNR_date = '25Apr24'

#There are 8 cpus available on my personal computer
num_cpus_avail = 60

if not add_noise:
    iterations = 1
    SNR_oi = np.nan
else:
    SNR_oi = 75
    iterations = 20 #Check that this is at most the largest of the data set

############## Setting Files Data ##########

cwd_temp = os.getcwd()
base_file = 'Regularization'
cwd_full = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

slice_num = 3
pat_id = "BLSA_1935_06_MCIAD_m79"#"BLSA_1742_04_MCIAD_m41"#"BLSA_1935_06_MCIAD_m79"

pat_tag = pat_id[-3:] #this tag will show up in the file name

output_folder = f"ExperimentalSets/{pat_id}"

if add_noise:
    noise_iter_folder = f'Noise_Generation/Noise_Sets/{pat_tag}_slice{slice_num}_SNR{SNR_oi}_{date_of_data}'
else:
    noise_iter_folder = f'MB_References/{pat_id}'
SNR_info_folder = f'MB_References/{pat_id}/SNR_info_{SNR_date}.pkl'

############# Global Parameters ###############

ob_weight = 100

if multistart_method:
    num_multistarts = 2
else:
    num_multistarts = 1

############# SNR Information ###############

file_path = f'{cwd_full}{SNR_info_folder}'
if os.path.isfile(file_path):
    print('Data was loaded in')
    with open(f'{cwd_full}{SNR_info_folder}', 'rb') as handle:
        dict = pickle.load(handle)
        mask_amplitude = dict['mask_amplitude'] 
        n_vert = dict['n_vert']
        n_hori = dict['n_hori']
        n_elem = dict['n_elem']
        handle.close()
else:
    raise ValueError(f'There is not a valid file to load. Check path:{file_path}')

file_path = f'{cwd_full}{noise_iter_folder}/hprParameter.pkl'
if add_noise:
    if os.path.isfile(file_path):
        with open(f'{cwd_full}{noise_iter_folder}/hprParameter.pkl', 'rb') as handle:
            dict = pickle.load(handle)
            SNR_desired = dict["SNR_desired"]
            n_noise_realizations = dict["num_noise_realizations"]
            SNR_root_folder = dict["SNR_file_source"]
            mask_shape = dict['mask_shape']
            handle.close()
    else:
        raise ValueError(f'There is not a valid file to load. Check path:{file_path}')

    assert(iterations <= n_noise_realizations)
    assert(SNR_root_folder == SNR_info_folder)
    assert(SNR_oi == SNR_desired)

t_increment_brain = 11.3 #This is a measurement originally indicated by Chuan Bi in the initial email about this data
TDATA = np.linspace(t_increment_brain, (n_elem)*(t_increment_brain), n_elem)

#This is how we will keep track of all voxels that are called
target_iterator = np.array([item for item in itertools.product(np.arange(0,n_vert,1), np.arange(0,n_hori,1))])

####### File Naming Section #########

date = date.today()
day = date.strftime('%d')
month = date.strftime('%B')[0:3]
year = date.strftime('%y')

seriesTag = f"{pat_tag}_slice{slice_num}_"
if add_noise:
    seriesTag = (seriesTag + f"SNR_{SNR_oi}_")
else:
    seriesTag = (seriesTag + f"NoNoise_")

if not apply_normalizer:
    seriesTag = (seriesTag + "NoNorm" + "_")

if model_selection:
    seriesTag = (seriesTag + "BIC_filter" + "_")

seriesTag = (seriesTag + addTag + day + month + year)

seriesFolder = (f'{cwd_full}/{output_folder}/{seriesTag}')
os.makedirs(seriesFolder, exist_ok = True)

############# Signal Functions ##############

def G_biX_off(t, con_1, con_2, tau_1, tau_2, offSet): 
    signal = con_1*np.exp(-t/tau_1) + con_2*np.exp(-t/tau_2) + offSet
    return signal

def G_moX_off(t, con, tau, offSet): 
    signal = con*np.exp(-t/tau) + offSet
    return signal



def G_reg(lam, func, SA = 1):
    #SA defines the signal amplitude, defaults to 1 with assumed normalized data
    #Regularization is only applied to biexponential data
    f_name = func.__name__
    if 'biX' in f_name:
        def Gt_lam(t, con_1, con_2, tau_1, tau_2, offSet):
            param_stack = [lam*con_1/SA, lam*con_2/SA, lam*tau_1/ob_weight, lam*tau_2/ob_weight]
            return np.append(G_biX_off(t, con_1, con_2, tau_1, tau_2, offSet), param_stack)
    else:
        raise Exception("Not a valid function: " + f_name)
    return Gt_lam

def G_reg_param(lam, func, popt, SA = 1):
    #SA defines the signal amplitude, defaults to 1 with assumed normalized data
    #Regularization is only applied to biexponential data
    f_name = func.__name__
    if 'biX' in f_name:
        param_stack = popt[:4]*np.array([lam/SA, lam/SA, lam/ob_weight, lam/ob_weight])
    else:
        raise Exception("Not a valid function: " + f_name)
    return param_stack

############# Selecting Function ###############

model_oi = G_biX_off

############# Data Processing Functions ##############

def mask_data(data, mask_amp):
    #Sets every decay curve in the data set where the amplitude is less than a threshold value to zero
    I_masked = np.copy(data)
    mask = I_masked[:,:,0]<mask_amp
    I_masked[mask] = 0
    return I_masked, mask

def normalize_brain(I_data):
    n_vert, n_hori, n_elem = I_data.shape
    I_normalized = np.zeros(I_data.shape)
    for i_vert in range(n_vert):
        for i_hori in range(n_hori):
            data = I_data[i_vert,i_hori,:]
            if data[0]>0:
                data_normalized = data/(data[0])
            else:
                data_normalized = np.zeros(n_elem)
            I_normalized[i_vert,i_hori,:] = data_normalized
    return I_normalized


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

    if (popt[-2+num] > popt[-1+num]): #We want by convention to make sure that T21 is <= T22
        for i in range(popt.shape[0]//2):
            p_hold = popt[2*i]
            popt[2*i] = popt[2*i+1]
            popt[2*i+1] = p_hold
    return popt

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

def calculate_BIC(RSS, popt):

    BIC = len(TDATA) * np.log(RSS/(len(TDATA)-len(popt)-1)) + (len(popt)+1)*np.log(len(TDATA))

    return BIC

################## Parameter Estimation Functions ###############

def single_reg_param_est(data, lam, func):
            
    init_p = get_param_p0(func, sig_init = data[0])
    upper_bound = get_upperBound(func)
    lower_bound = np.zeros(len(upper_bound))
    parameter_tail = np.zeros(len(upper_bound)-1)
    data_tilde = np.append(data, parameter_tail) # Adds zeros to the end of the regularization array for the param estimation
    
    try:
        popt, _, info, _, _ = curve_fit(G_reg(lam, func, SA = data_tilde[0]), TDATA, data_tilde, bounds = (lower_bound, upper_bound), p0=init_p, max_nfev = 4000, full_output = True)
    except Exception as error:
        popt = [0,0,1,1,0]
        info['fvec'] = np.inf
        print("Error in parameter fitting: " + str(error))

    RSS = np.sum(info['fvec']**2)

    return popt, RSS

def perform_multi_estimates(data, lam, func, n_initials = num_multistarts):
    #Pick n_initials random initial conditions within the bound, and choose the one giving the lowest model-data mismatch residual

    RSS_hold = np.inf
    reg_RSS_hold = np.inf
    for i in range(n_initials):

        popt, reg_RSS_temp = single_reg_param_est(data, lam, func)

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

    RSS_biX = calculate_RSS(data, G_biX_off_params, G_biX_off)
    RSS_moX = calculate_RSS(data, G_moX_off_params, G_moX_off)

    BIC_G_biX = calculate_BIC(RSS_biX, G_biX_off_params)
    BIC_G_moX = calculate_BIC(RSS_moX, G_moX_off_params)

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


#### Looping through Iterations of the brain - applying parallel processing to improve the speed
if __name__ == '__main__':
    freeze_support()

    for iter in range(iterations):    #Build {iterations} number of noisey brain realizations

        np.random.seed(iter)

        if add_noise:
            filename = f'{cwd_full}{noise_iter_folder}/{pat_tag}_slice{slice_num}_iter{iter+1}.pkl'
            fileObject = open(filename, 'rb')
            I_noised = pickle.load(fileObject)
            fileObject.close()
        else:
            brain_data = scipy.io.loadmat(f'{cwd_full}{noise_iter_folder}/NESMA_cropped_slice{slice_num}.mat')
            I_noised, mask = mask_data(brain_data["slice_oi"], mask_amplitude)
            mask_shape = (mask+1)%2

        if apply_normalizer:
            noise_iteration = normalize_brain(I_noised)
        else:
            noise_iteration = I_noised

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
    "SNR_oi": SNR_oi,
    'n_noise_realizations': iterations,
    'mask_shape': mask_shape,
    'lambdas': lambdas,
    'SNR_info_folder': SNR_info_folder,
    "noise_iter_folder": noise_iter_folder,
    "data_slice": slice_num,
    "pat_id": pat_id,
    'tdata': TDATA,
    'ob_weight': ob_weight,
    'num_multistarts': num_multistarts,
    'model_oi': model_oi,
    'upper_bound': get_upperBound(model_oi),
    'options': [add_noise, apply_normalizer, 
                model_selection, multistart_method]
}

f = open(seriesFolder + '/hprParameter_info_' + day + month + year +'.pkl','wb')
pickle.dump(hprParams,f)
f.close()