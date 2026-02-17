# --------------------------------------------------------------------
# Divide and Conquer Algorithm to find the proximal operator of the L-infinity norm:
#   prox(x) = argmin_y [ (1/2)||y-x||_2^2 + alpha*||y||_inf ] 
# 
# Input:
#   x: data (vector)
#   alpha: constant on penalty term
#    
# Output: prox(x) 
# --------------------------------------------------------------------

import numpy as np

def prox_op_dc(x, alpha, seed):

    np.random.seed(seed)
 
    if alpha >= np.linalg.norm(x,1):
        tau = 0
        prox = np.zeros(x.shape)
    else:
        x_hat = np.abs(x[x != 0])

        # initializations
        l_max = 0; nu_tilde = 0; n_nu_tilde = 0

        while len(x_hat) != 0:
            
            # select a pivot 
            p = np.random.choice(x_hat)
            #print(p)

            # partition x_hat based on the pivot          
            u = x_hat[x_hat > p]
            n_nu = len(u)
            nu = np.sum(u)

            # find psi'(p)
            deriv = -(nu_tilde+nu) + (n_nu_tilde+n_nu)*p + alpha

            # update l and u
            if deriv < 0:
                l_max = p
                x_hat = u
            else:
                u_min = p
                n_p = np.sum(x_hat == p)
                nu_tilde += (nu + (p*n_p))  # sumE = p*n_p
                n_nu_tilde += (n_nu + n_p)
                x_hat = x_hat[x_hat < p]
            
        # end while

        # find minimizer (will be in (l_max, u_min] for tau != 0) 
        tau = (nu_tilde - alpha)/n_nu_tilde

        # compute proximal operator
        prox = x.copy()
        idx = (np.abs(x) > tau)
        prox[idx] = np.sign(x[idx])*tau
            
    return prox, tau

