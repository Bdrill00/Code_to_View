import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("./simple_pdhg_tests")

from create_A_matrix_4bus_Qineq import *

A, b, x_names = get_power_system_matrices()

mm, nn = A.shape
c = np.zeros(nn)
c[8] = 1
c[9] = 1

# Let's also build C and d: Cx le d
C = np.zeros((2,nn))
C[0,16] = 1
C[1,16] = -1
d = np.array([+1.1, -0.9])

# now, apply pdhg
x = np.ones(A.shape[1]); lam = np.zeros(A.shape[0]); mu = np.zeros(2)
nsteps = 5000000; eta = 0.000025; n_list = np.zeros(nsteps)
for i in range(nsteps):

    reg1 = 10.0*(2*(A@x-b).T@A).T
    x   = x - eta*(c + A.T@lam +C.T@mu) - eta*reg1
    
    reg2l = 10.0*2*(c+A.T@lam + C.T@mu).T@(A.T)
    reg2m = 10.0*2*(c+A.T@lam + C.T@mu).T@(C.T)
    lam = lam + eta*(A@x - b) - eta*reg2l
    mu  = mu  + eta*(C@x - d) - eta*reg2m
    
    x[16] = np.maximum(0.9,x[16])
    x[16] = np.minimum(1.1,x[16])
    mu  = np.maximum(mu, 0)
    
    n_list[i] = np.linalg.norm(A@x-b) + np.linalg.norm(c + A.T@lam + C.T@mu)
    print(n_list[i])
print()

print(f"V1: {(x[0]):.4f} + j{(x[1]):.4f}")
print(f"V2: {(x[2]):.4f} + j{(x[3]):.4f}")
print(f"V3: {(x[6]):.4f} + j{(x[7]):.4f}")
print(f"V4: {(x[8]):.4f} + j{(x[9]):.4f}")
print(f"I_slack: {(x[10]):.4f} + j{(x[11]):.4f}")
print(f"Vset: {(x[16]):.4f}")
    
plt.close() 
plt.plot(n_list)
plt.yscale("log")
plt.savefig("norms.png")