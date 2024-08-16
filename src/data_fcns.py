import numpy as np
import time

from prox_op import prox_op

def generate_raw_data(data_dist="both", min_len=10, max_len=100, num_vec=10000, unif_min=0, unif_max=1, seed=1):

    # CREATE RAW DATA ---------------------------------------
    #
    #  - X: num_vec x max_len, each observation is a row
    #  - lengths: vector lengths for each row of X
    #  - alpha: for each prox op problem
    #  - tau: calculated from prox op 
    
    np.random.seed(seed)
    
    if data_dist == "norm":
        X = np.random.randn(num_vec, max_len) # Gaussian
    elif data_dist == "unif": 
        X = np.random.uniform(unif_min, unif_max, (num_vec, max_len) ) # Uniform [unif_min, unif_max)
    else: # both normal and uniform data
        Xn = np.random.randn(int(num_vec/2), max_len) # Gaussian
        Xu = np.random.uniform(unif_min, unif_max, (int(num_vec/2), max_len) ) # Uniform
        X = np.vstack((Xn,Xu))
    
    lengths = np.random.randint(low=min_len, high=(max_len+1), size=num_vec) 

    # NOTE: rand - [0,1), so alpha between [1,6).  This is done to prevent division by small (near zero) alphas
    # that will cause training set entries to blow up.
    alphas = 5*np.random.rand(num_vec) + 1 
    
    # compute tau for each vector (threshold from prox op)
    t1 = time.perf_counter()

    taus = np.zeros(num_vec)
    for i in range(num_vec):
        
        if i%1000 == 0:
            print(i)
        
        p, taus[i], istar = prox_op(X[i,0:lengths[i]], alphas[i])
    
    t2 = time.perf_counter()
    print(f'Time for tau calculations: {t2-t1}')     
    
    return X, lengths, alphas, taus
    

def vanilla_scaling(X, lengths, alphas, taus):

    num_obs, max_len = X.shape
    M = np.zeros((num_obs, max_len))
    yhat = np.zeros(num_obs)
    zero_idx = np.zeros(num_obs, dtype=bool)
    
    for i in range(num_obs):
        
        #if i%1000 == 0:
        #    print(i)
        
        len_v = lengths[i]
        x = X[i,0:len_v] 
        alpha = alphas[i]
    
        w = np.abs(x)/alpha
        w_1norm = np.linalg.norm(w,1) 

        if w_1norm > 1:
            
            M[i,0:len_v] = w

            # transform y (tau) 
            yhat[i] = taus[i]/alpha
            
        else:
            print(f'Zero index {i}')
            zero_idx[i] = True
    
    return M, yhat, zero_idx


def compute_features(X, lengths, alphas, taus, num_moments):

    num_obs = X.shape[0]
    M = np.zeros((num_obs, num_moments+3))
    yhat = np.zeros(num_obs)
    zero_idx = np.zeros(num_obs, dtype=bool)
    mus = np.zeros(num_obs)
    
    for i in range(num_obs):
        
        #if i%1000 == 0:
        #    print(i)
        
        len_v = lengths[i]
        x = X[i,0:len_v]
        alpha = alphas[i]
        
        w = np.abs(x)/alpha
        w_1norm = np.linalg.norm(w,1) 

        if w_1norm > 1:

            mu = w_1norm/len_v
            v = w - mu 

            m = np.zeros(num_moments+3) # min, max, moments, length
            m[0] = np.min(v)
            m[1] = np.max(v)
            m[2] = np.linalg.norm(v,1)/len_v  # L1  

            # second moment: sum(x^2)
            v_power = np.square(v)
            m[3] = np.sqrt( np.sum(v_power)/len_v )

            # jth moment: sum(x^i)
            for j in range(3, num_moments+1): 
                v_power = v_power*v  # v^j
                mom = np.sum(v_power)/len_v
                if j % 2 == 1: # odd moment      
                    m[j+1] = np.sign(mom)*np.power(abs(mom), 1/j)
                else: # even moment
                    m[j+1] = np.power(mom, 1/j)

            m[2+num_moments] = np.log(len_v)

            M[i,:] = m
            
            # transform y (tau) 
            yhat[i] = (taus[i]/alpha) - mu
            
            mus[i] = mu
        
        else:
            print(f'Zero index {i}')
            zero_idx[i] = True

    return M, yhat, mus, zero_idx    
                    