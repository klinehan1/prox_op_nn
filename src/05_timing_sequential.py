# # Timing: Comparsions for Exact Prox Op and NN Prox Op Calculations
# 
# Comparison of:
# - sorting-based exact prox op algorithm
# - divide and conquer exact prox op algorithm
# - NN prox op approximation (includes data processing, NN inference, and prox op calculation)
# 
# 
# Timings are done:  
# - sequentially (one vector processed at a time in a loop)
#
# These are the official timings reported. All timings are done on a multicore CPU.
#
# Usage with PyTorch 1.12.0 container: 
# apptainer exec ~/pytorch-1.12.0.sif python 04_timing_sequential.py vec_len num_mom


# # System Set-Up --------------------------------------

import numpy as np
from matplotlib import pyplot as plt 
import time as time
import sys
import platform, psutil
from numba import jit, prange

import torch
from torch import nn

from prox_op import prox_op
from prox_op_dc import prox_op_dc
from data_fcns import generate_raw_data, vanilla_preprocess, compute_features

vec_len = int(sys.argv[1])
num_mom = int(sys.argv[2])
print(f"Args: vec_len={vec_len}, num_mom={num_mom}")

# python version
print(sys.version)

# get CPU info
print(platform.processor())
print(platform.machine())
print(platform.version())
print(platform.platform())
print(platform.uname())
print(platform.system())
print(str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB")

# get GPU info
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

# # Generate Raw Data -----------------------------------------

nn_type = "feature"  # vanilla or feature
data_dist = "both"   # norm, unif, or both
unif_min = 0
unif_max = 1
min_len = vec_len
max_len = vec_len
num_vec = 10000
seed = 1
num_moments = num_mom

X, lengths, alphas, taus = generate_raw_data(data_dist, min_len, max_len, num_vec, unif_min, unif_max, seed)

# # Exact Prox Op, Sorting-Based Algorithm ----------------------------------------------------

print(f"Sequential Timings:")
print(f"---------------")
# Sequential

t1 = time.perf_counter()

PROX_OG = np.zeros(X.shape)
for i in range(PROX_OG.shape[0]):
    PROX_OG[i,:] = prox_op(X[i,:], alphas[i])[0]

t2 = time.perf_counter()    

print(f"Exact, Sorting-Based:")
print(f"Total Time: {t2-t1}")
print(f"Average Time per Vector: {(t2-t1)/X.shape[0]}")
print(f"---------------")

del PROX_OG

# ## Exact Prox Op, Divide and Conquer Algorithm ---------------------------------------------

# ### Sequential

t1 = time.perf_counter()

PROX_OG = np.zeros(X.shape)
for i in range(PROX_OG.shape[0]):
    PROX_OG[i,:] = prox_op_dc(X[i,:], alphas[i], seed)[0]

t2 = time.perf_counter()    

print(f"Exact, DC:")
print(f"DC Total Time: {t2-t1}")
print(f"DC Average Time per Vector: {(t2-t1)/X.shape[0]}")
print(f"---------------")

del PROX_OG

# # Features NN ------------------------------------------
# 
# Assume that NN is already loaded on CPU before timings.

# ### Load NN
np.random.seed(0)
torch.manual_seed(0)
device = "cpu"

# features NN

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_moments+3, 25),  
            nn.ReLU(),    
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        tau = self.linear_relu_stack(x) 
        return tau

model = NeuralNetwork().to(device)
print(model)

# load model

if num_mom == 5:
    if vec_len == 1000:
        model.load_state_dict(torch.load("/home/kjl5t/L_inf_prox_op/src/models/features_5m/both/len_1000_2000/epoch_4370_nn.pt"))
    else:  # vec_len = 10,000 or 100,000
        model.load_state_dict(torch.load("/home/kjl5t/L_inf_prox_op/src/models/features_5m/both/len_1000_100000/epoch_4773_nn.pt"))

else: # num_mom = 10
    if vec_len == 1000:
        model.load_state_dict(torch.load("/home/kjl5t/L_inf_prox_op/src/models/features/both/len_1000_2000/epoch_4570_nn.pt"))
    else:  # vec_len = 10,000 or 100,000
        model.load_state_dict(torch.load("/home/kjl5t/L_inf_prox_op/src/models/features/both/len_1000_100000/epoch_4792_nn.pt"))

model.eval()

# ## Compute Features - Sequential 
t1 = time.perf_counter()
M2, yhat2, mus2, zero_idx2 = compute_features(X, lengths, alphas, taus, num_moments)
t2 = time.perf_counter()

print(f"NN, Compute Features:")
print(f"Total Time: {t2-t1}")
print(f"Average Time per Vector: {(t2-t1)/M2.shape[0]}")

t_comp_features = t2-t1

#  check if any vectors have tau=0
print(f"Number of vectors with tau=0: {sum(zero_idx2)}")

# ### NN Inference - CPU, NOT batched, sequential

t1 = time.perf_counter()

pred_tau_hat2 = np.zeros(M2.shape[0])
for i in range(len(pred_tau_hat2)):
    with torch.inference_mode():
        pred_tau_hat2[i] = model(torch.Tensor(M2[i,:]))    

t2 = time.perf_counter()    

print(f"NN, Inference:")
print(f"Total Time: {t2-t1}")
print(f"Average Time per Vector: {(t2-t1)/M2.shape[0]}")

t_nn_inf = t2-t1

# ### Compute Prox Op with Predicted Taus - Sequential

t1 = time.perf_counter()  

PROX = np.copy(X)

for i in range(PROX.shape[0]):

    pred_tau = alphas[i]*(pred_tau_hat2[i]+mus2[i])
    idx = (np.abs(X[i,:]) > pred_tau)
    PROX[i,idx] = np.sign(X[i,idx])*pred_tau

t2 = time.perf_counter()

print(f"NN, Compute Prox Op:")
print(f"Total Time: {t2-t1}")
print(f"Average Time per Vector: {(t2-t1)/X.shape[0]}")

t_prox_op = t2-t1

# ### Calulate total time

t_total = t_comp_features + t_nn_inf + t_prox_op

print(f"NN, Total Time:")
print(f"Total Time: {t_total}")
print(f"Average Time per Vector: {t_total/M2.shape[0]}")
print(f"---------------")


