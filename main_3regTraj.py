import csv

from itertools import product, zip_longest
import multiprocessing

import numpy as np

import os

import pandas as pd

import sys

import torch

from tqdm import tqdm, trange

# from Reordering_Swapping_for_GPU import parameter_swap_same_tensor

# from makeSignals import myTrueModel, myNoisyModel, myTrueModel_2param

# from regTraj import least_squares_2param, least_squares_3param, curve_fit_2param, make_l2_trajectory, curve_fit_l2Regularized_2param

# from writeParams import writeSummary

import multiprocess as mp
from multiprocess import RawArray

from multiprocessing import Pool, freeze_support
from multiprocessing import set_start_method
import torch.multiprocessing as mpp

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
from DF_DataLoader import initDataset
from scipy.optimize import curve_fit
import ray
from ray.util.multiprocessing import Pool
import functools

def mycurvefit_l2Regularized_3param(i, datatype = None, signalType="biexponential", lb_T21=0.0, lb_T22=0.0, lb_c1=0.0, ub_T21=np.inf, ub_T22=np.inf, ub_c1=np.inf):
        
        dataPurpose = datatype


#         if dataPurpose == "Testing":
#             T21_low, T21_high = 75.0, 475.0  # lower, upper bound on T2,1

#             num_T22, T22_low, T22_high = 200, 75.0, 475.0

#         elif dataPurpose == "Validation":
#             T21_low, T21_high = 75.0, 475.0  # lower, upper bound on T2,1

#             num_T22, T22_low, T22_high = 200, 75.0, 475.0

#         elif dataPurpose == "Training":
#             T21_low, T21_high = 50.0, 500.0  # lower, upper bound on T2,1

#             num_T22, T22_low, T22_high = 200, 50.0, 500.0

#         time_low, time_high = 0.0, 1.6 * T22_high

        # print(T22_high)

        times = np.arange(11.3, 11.3*129, 11.3, dtype=np.float64)
        # np.linspace(time_low, time_high, 64, )

        D = np.array([1.0,  # T2,1

                    1.0,  # T2,2

                    1.0])  # C1
        # D = np.array([1.0, 1.0])
        

        # j = data[i][0].int().item()
        ld = data[i][0]
        c1 = data[i][1]
        c2 = data[i][2]
        p_0 = data[i][3:6]
        
        d = data[i][6:]
        # print(len(d))  
        assert(len(d) == 128)
        # times = times_test
        # D = D_w
        """

        Approximates the solution of the L^2 regularized least-squares problem:



        (*)        argmin_{p in R^2} (||G(p,c) - d||_2^2 + (ld**2)*||D*p||_2^2)



        where G : R^2 -> R^(num_times) == an operator that maps the pair of

        parameters (T21, T22) to a discrete signal measured at the input times.



        Input:

        ------

            1. d (Numpy array of length len(times)) - noisy signal

            2. times (numpy array) - times at which the signal == measured

            3. ld (float) - regularization parameter

            4. p_0 (Array of length 3) - Initial guess of solution for the NLLS solver

            5. signalType (string, optional kwarg) - Type of signal. Choices are

                biexponential, power, quadratic, or sinusoidal.

            6. lb_T21 (float) - Lower T21 bound - set to 0.0

            7. lb_T22 (float) - Lower T22 bound - set to 0.0

            8. ub_T21 (float) - Upper T21 bound - set to inf

            9. ub_T22 (float) - Upper T22 bound - set to inf



        Output:

        -------

            1. p (Array of length 2) - The parameter estimates, i.e., the approximate

                solution to (*).

        """



        # A) Define curve to fit data to. 

        #    Needs to have a call signature (xdata, parameter

        def signal(xdata, p1, p2, p3):



            if signalType == "biexponential":



                # 1. number of signal acquisition times

                num_times = len(xdata) - 1



                # 2. extract data

                t_vec = xdata[0:num_times]   # times

                reg_param = xdata[num_times] # regularization parameter



                # 3. calculate penalty

                params = np.array([p1, p2, p3], dtype=np.float64)

                penalty_vec = ld*np.multiply(D,params)



                # 4. concatenate and return

                return np.concatenate((p3*np.exp(-t_vec/p1) + (1.0-p3)*np.exp(-t_vec/p2),

                                    penalty_vec))

                                    



        # B) Bounds on the parameters

        #lb_T21, ub_T21 = 0.0, np.inf  # lb can be small & positive to enforce nonnegativity

        #lb_T22, ub_T22 = 0.0, np.inf



        # C) Fit given dependent variable to curve and independent variable

        # Uses 2 point finite-differences to approximate the gradient.

        # Could also use 3 point or take exact gradient.

        t_dim = times.ndim

        indep_var = np.concatenate((times,

                                    np.array(ld,ndmin=t_dim)))



        d_dim = d.ndim

        depen_var = np.concatenate((d, np.array(0.0, ndmin=d_dim), np.array(0.0,ndmin=d_dim), np.array(0.0, ndmin=d_dim)))

        
        try:
            opt_val = curve_fit(signal, indep_var, depen_var,  # curve, xdata, ydata

                                p0=p_0,  # initial guess

                                bounds=([lb_T21, lb_T22, lb_c1], [ub_T21, ub_T22, ub_c1]),

                                method="trf",

                                max_nfev=1000)
            #print('!!!!!!!!!!', opt_val)
        except RuntimeError:
            opt_val = (np.asarray([66.08067015, 66.44472936, 45.5891436 ]), np.asarray([[ 6.51065099e+03, -6.53878183e+03,  1.61616729e+06], [-6.53878183e+03,  6.85069747e+03, -1.65784453e+06], [ 1.61616729e+06, -1.65784453e+06,  4.05431645e+08]]))
            
            print("maximum number of function evaluations == exceeded")





        # returns estimate. second index in estimated covariance matrix

        T21_ld, T22_ld, c1_ret = opt_val[0]
    # Enforces T21 <= T22
    # T21_ld = np.where(T21_ld > T22_ld, T22_ld, T21_ld)
        if T21_ld.size == 1:
            T21_ld = T21_ld.item()
            T22_ld = T22_ld.item()
            c1_ret = c1_ret.item()
        
        if T21_ld > T22_ld:
            T21_ld_new = T22_ld
            T22_ld = T21_ld
            T21_ld = T21_ld_new
            c1_ret = 1.0 - c1_ret
            assert (T21_ld != T22_ld)


        # return opt_val[0]
        if i == None:
            return d, ld, c1_ret, T21_ld, T22_ld 
        else:
            return d, ld, c1_ret, T21_ld, T22_ld, i


for dp in ["Training", "Validation", "Testing"]:

    curr_path = os.path.abspath('')
    # training_path = os.path.relpath("../Lambda_TrainingData/sizetestfminBound_LambdaTraining_SNR_900.0_TrainingData.feather", curr_path)

    convolutional = True
    # training_dataset = initDataset(training_path, set_type = "training", type1 = "standardized", convolutional = convolutional)

    training_path_2 = os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/3P_LambdaTraining_Myelin_128Pts_SNR_900.0_{dp}Data.feather", curr_path)

    training_dataset_2 = initDataset(training_path_2, set_type = "validation", select_target = ["T21_t", "T22_t", "c1_t"], type1 = "standardized", mean= torch.tensor([0]), std = torch.tensor([1]), convolutional = convolutional)



    #TEST INITIALIZER WITH POOL BY DOING NP.RANDOM


    noisy_signals = (training_dataset_2.training_tensor_proc).squeeze(1)
#     for option in [0,1]:
#         if option == 0:
#             lambdas_t2s = torch.full((noisy_signals.shape[0],), 0.0) #training_preds #
#         elif option == 1:
    lambdas_t2s = torch.load(os.path.relpath(f"../Lambda_TrainingData/LambdaGeneration/3PE_Myelin_128Pts_lambdas_{dp}Data.pt", curr_path))
#         lambdas_t2s = training_preds


    c1_t2s = training_dataset_2.master_frame["c1_t"]
    c2_t2s = 1.0 - training_dataset_2.master_frame["c1_t"]
    p_s = training_dataset_2.target_tensor_proc


    # mpp.set_start_method('spawn', force=  True) #Artifact -- may not be needed anymore
    print("LAMBDA NANs: ", np.where(np.isnan(lambdas_t2s)))
    print("LAMBDA Infs: ", np.where(np.isinf(lambdas_t2s)))
    print("ND NANs: ", np.where(np.isnan(noisy_signals)))
    print("ND Infs: ", np.where(np.isinf(noisy_signals)))


    # T2_ld = torch.empty((noisy_signals.shape[0], 2))
    j_NDs = torch.cat((lambdas_t2s.view(-1,1),
                        torch.tensor(c1_t2s.values).view(-1,1), torch.tensor(c2_t2s.values).view(-1,1),
                        p_s,
                        noisy_signals), dim = 1)


    print("Starting MP...")

    # lis = []



    data = j_NDs.detach().numpy()


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

        print(f"Building {dp} Dataset...")
        lis = []

        with mp.Pool(processes = num_cpus_avail) as pool:

            with tqdm(total=data.shape[0]) as pbar:
                for ND, ld, c1, t21, t22, k in pool.imap_unordered(
                    functools.partial(mycurvefit_l2Regularized_3param, datatype=dp), range(data.shape[0]), chunksize = 250):
                    # if k == 0:
                        # print("Starting...")

                    lis.append(np.concatenate([np.array([c1, t21, t22, k, ld]), ND], dtype=np.float64))

                    pbar.update()
        #             # break
            pool.close() #figure out how to implement
            pool.join()
        # # assert False
        T2_ld = np.stack([row for row in lis], axis = 0)
        print(T2_ld.shape) #should be num_triplets X num_realizations
        # assert(T2_ld.shape == )
        df = pd.DataFrame(index = range(T2_ld.shape[0]), columns = ["c1_ld","t21_ld", "t22_ld", "Index", "lambda", "ND"])
        df[["c1_ld", "t21_ld", "t22_ld", "Index", "lambda"]] = T2_ld[:,:-128]
        df["ND"] = [T2_ld[i,-128:] for i in range(T2_ld.shape[0])]
        print("DATAFRAME: ", df.shape)
        # df[df.columns] = T2_ld
        # df.set_index("Index", inplace=True, drop = True)
        # df.sort_index(inplace = True)
#             if option == 0:
#                 df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_Lambda0_SNR_900.0_{dp}Data.feather")
#             elif option ==1:
        df.to_feather(f"../Lambda_TrainingData/3PE_ReconSignals/3PE_ReconSignals_LambdaNN_Myelin_128Pts_SNR_900.0_{dp}Data.feather")               