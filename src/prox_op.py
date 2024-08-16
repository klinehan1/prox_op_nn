# --------------------------------------------------------------------
# Algorithm to find the proximal operator of the L-infinity norm:
#   prox(x) = argmin_y [ (1/2)||y-x||_2^2 + alpha*||y||_inf ] 
# 
# Input:
#   x: data (vector)
#   alpha: constant on penalty term
#    
# Output: prox(x) 
# --------------------------------------------------------------------

import numpy as np

def prox_op(x, alpha):
    
    m = len(x)
    
    if alpha >= np.linalg.norm(x,1):
        tstar = 0
        istar = m
        prox = np.zeros(x.shape)
    else:
        # permute x to be in decreasing order in abs value
        s = np.sort(np.abs(x))[::-1]
        s = np.append(s,0)
    
        # find value for minimizer    
        tstar = 0
        istar = m
        s_sum = 0
        i = 0
        while i < m:  # len(x) = m
            s_i = s[i]
            s_sum = s_sum + s_i
            
            # check for repeated elements
            j = 1
            while (i+j < m) and s[i+j] == s_i:  
                s_sum = s_sum + s_i
                j = j+1
            
            i = i + (j-1)

            t0 = (s_sum - alpha)/(i+1)  # minimizer

            if (t0 <= s[i]) and (t0 > s[i+1]): 
                tstar = t0
                istar = i+1
                break

            i = i+1
        # end while
        
        # compute proximal operator
        prox = x.copy()
        idx = (np.abs(x) > tstar)
        prox[idx] = np.sign(x[idx])*tstar
            
    return prox, tstar, istar