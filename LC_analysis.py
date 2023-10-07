import numpy as np
import mpmath
import scipy.constants as const
import matplotlib.pyplot as plt
import time

#%% capacitor
epsilon_r = 11.9
epsilon_eff = (1+epsilon_r)/2 * const.epsilon_0

def Cal_C_in(w, s, h):
    eta = w/(w+s)
    k = np.sin(np.pi*eta/2)
    k_prime = np.sqrt(1-k**2)
    C_in = (2*epsilon_eff) * h * mpmath.ellipk(k**2)/mpmath.ellipk(k_prime**2)
    return C_in

def Cal_C_out(w, se, we, h):
    k = (w/2)/(w/2+se) * np.sqrt((w/2+se+we)**2-(w/2+se)**2)/np.sqrt((w/2+se+we)**2-(w/2)**2)
    k_prime = np.sqrt(1-k**2)
    C_out = (2*epsilon_eff) * h * mpmath.ellipk(k**2)/mpmath.ellipk(k_prime**2)
    return C_out

#%% inductor
def Cal_L_self(w,l):
    x = w/l
    if l != 0:
        L_self = (2e-7*l)*(np.log(2/x)+x/3+0.5)
    else:
        L_self = 0
    return L_self

def Cal_L_mutual(d,l, w=1.0,w1=1.0,GMD=False):
    if l != 0:
        if GMD:
            d_g = Cal_d_g(d, w, w1)
            x = d_g/l
        else:
            x = d/l
        Q = np.log(1+np.sqrt(1+x**2)) - np.log(x) + (x-np.sqrt(1+x**2))
        L_mutual = (2e-7*l)*Q
    else:
        L_mutual = 0
    return L_mutual

def Cal_d_g(d, w, w1):
    y = np.log(d) - (3/2)\
        + 1/(2*w*w1) * (d+(w+w1)/2)**2 * np.log(1+(w+w1)/(2*d))\
        + 1/(2*w*w1) * (d-(w+w1)/2)**2 * np.log(1-(w+w1)/(2*d))\
        - 1/(2*w*w1) * (d+(w-w1)/2)**2 * np.log(1+(w-w1)/(2*d))\
        - 1/(2*w*w1) * (d-(w-w1)/2)**2 * np.log(1-(w-w1)/(2*d))
    return np.exp(y)

#%% simulation parameters
w_ind = 2e-6
d_ind = 25 * 1e-6
h_ind = 250e-6
N_ind = 10

w_cap = 10e-6
s_cap= 2e-6
h_cap = 200 * 1e-6
N_cap = 50

#%% self inductance
t = time.time()

L_hori = N_ind * Cal_L_self(w_ind,d_ind+w_ind)
L_vert = (N_ind+1) * Cal_L_self(w_ind,h_ind)
L_self = L_hori + L_vert

#%% mutual inductance
L_mutual = 0
sgn_mutual = 1.0
N_mutual = N_ind

for ind_dis in range(0, N_ind):
    sgn_mutual = -1*sgn_mutual
    d_mutual = (ind_dis+1)*(d_ind+w_ind)
    N_mutual = N_ind - ind_dis
    
    if N_mutual > 0:
        L_mutual += 2*sgn_mutual * N_mutual * Cal_L_mutual(d_mutual,h_ind)

L_sum = L_self + L_mutual
T_ind = time.time() - t

#%% capacitance
t = time.time()

C1 = Cal_C_in(w_cap, s_cap, h_cap)
C2 = Cal_C_in(w_cap, s_cap, h_cap)
Ce = Cal_C_out(w_cap, s_cap, w_cap, h_cap)

C_sum = (N_cap-3)*C1*C2/(C1+C2) + 2*Ce
T_cap = time.time() - t

#%% 
w_0 = 1/np.sqrt(L_sum*C_sum)
Z_0 = np.sqrt(L_sum/C_sum)

print("Frequency: %f, Impedance: %f, Inductance: %f, Capacitance: %f" % (w_0/(2*np.pi)/1e9, Z_0, L_sum*1e9, C_sum*1e12))

