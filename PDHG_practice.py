import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# reproducibility
np.random.seed(1)

# data
c = np.random.randn(10)
A = np.random.randn(5, 10)
b = np.random.randn(5)
xl = -np.ones(10)
xu = np.ones(10)

# === Solve with cvxpy (Gurobi backend if available) ===
x = cp.Variable(10)
constraints = [A @ x == b,
               x >= xl,
               x <= xu]
objective = cp.Minimize(c @ x)
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI)

opt_val = prob.value
print("Optimal value (cvxpy):", opt_val)

# === Simple PDHG-like test ===
def f():
    nsteps = 10000
    xv = np.ones(10)*2
    lambd = np.zeros(5)
    eta = 5e-3
    Lag = np.zeros(nsteps)

    for ii in range(nsteps):
        eta *= 0.9999
        xv = xv - eta * (c + A.T @ lambd)
        xv = np.minimum(np.maximum(xv, xl), xu)
        lambd = lambd + eta * (A @ xv - b)
        Lag[ii] = c @ xv + lambd @ (A @ xv - b)
        print(Lag[ii])
    
    print()
    print("Optimal value (cvxpy):", opt_val)
    return Lag

Lag = f()

# === Plot ===
plt.plot(Lag, label="Lag iterates")
plt.plot(range(1, 10001), opt_val * np.ones(10000), label="Optimal value")
plt.legend()
plt.savefig("myplot3.png")
plt.show()