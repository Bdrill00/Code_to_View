import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_A_matrix_4bus import *

A, b, x_names = get_power_system_matrices()
x_solution = np.linalg.solve(A, b)

# now, apply pdhg
x = np.ones(A.shape[1]); lam = np.zeros(A.shape[0])
nsteps = 1000000; eta = 0.000025; n_list = np.zeros(nsteps)

for i in range(nsteps):

    reg1 = 10*(2*(A@x-b).T@A).T
    x   = x - eta*A.T@lam - eta*reg1
    
    reg2 = 10*2*(A.T@lam).T@A.T
    lam = lam + eta*(A@x - b) - eta*reg2
    
    n_list[i] = np.linalg.norm(A@x-b)
    print(n_list[i])
print()

plt.close() 
plt.plot(n_list)
plt.yscale("log")
plt.savefig("norms.png")