import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt  # added for plotting

key = jax.random.PRNGKey(1)

# Problem data
c = jax.random.normal(key, (10,))
A = jax.random.normal(key, (5, 10))
b = jax.random.normal(key, (5,))
xl = -jnp.ones(10)
xu =  jnp.ones(10)

def step_fun(ii, state):
        xv, lam, eta, Lag = state
        eta = 0.9999 * eta
        xv = xv - eta * (c + A.T @ lam)
        xv = jnp.clip(xv, xl, xu)
        lam = lam + eta * (A @ xv - b)
        lagval = c @ xv + lam @ (A @ xv - b)
        Lag = Lag.at[ii].set(lagval)
        return xv, lam, eta, Lag

def f(nsteps=100):
    xv = jnp.zeros(10)
    lam = jnp.zeros(5)
    eta = 5e-3
    Lag = jnp.zeros(nsteps)
    input = (xv, lam, eta, Lag)
    for i in range(0,nsteps):
        input = step_fun(i, input)
        if i == nsteps:
            return input
    xv, lam, eta, Lag = input
    return Lag, xv, lam

# Run the algorithm
Lag, xv, lam = f()
obj_val = c @ xv

# Print final results
print("Final objective value:", obj_val)
print("Final x:", xv)
print("Final dual variable λ:", lam)
print("Last Lagrangian value:", Lag[-1])

# ---- Convergence plot ----
plt.figure(figsize=(8,5))
plt.plot(range(1, len(Lag)+1), np.array(Lag), marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Lagrangian Value')
plt.title('Convergence of Lagrangian')
plt.grid(True)
plt.tight_layout()
# plt.savefig('/users/b/d/bdrillin/Practice/lagrangian_convergence.png')  # save figure to file
# print("Convergence plot saved as 'lagrangian_convergence.png'")