from Class_4_Bus import* #Line, Transformer, Generator, Load, powerflow
import math
import numpy as np
import jax
import jax.numpy as jnp
phases = ['a','b','c']
num_nodes = 4
generators = 1
loads = 1
vSlack = 12470 / jnp.sqrt(3)
variables = []  # combined variable list
positions = {}  # dictionary to store ranges

# Node voltages
start = len(variables)
for n in range(1, num_nodes+1):
    for p in phases:
        variables.append(f"v{n}r{p}")
    for p in phases:
        variables.append(f"v{n}i{p}")
end = len(variables)
positions['lines'] = (start, end)

# Generator currents
start = len(variables)
for n in range(1, generators+1):
    for p in phases:
        variables.append(f"Igr{p}")
    for p in phases:
        variables.append(f"Igi{p}")
end = len(variables)
positions['generators'] = (start, end)

# Load currents
start = len(variables)
for n in range(1, loads+1):
    for p in phases:
        variables.append(f"Ilr{p}")
    for p in phases:
        variables.append(f"Ili{p}")
end = len(variables)
positions['loads'] = (start, end)

#McCormick Variables
start = len(variables)
for n in range(1, loads +1):
    for p in phases:
        variables.append(f"xr{n}{p}")
    for p in phases:
        variables.append(f"yr{n}{p}")
    for p in phases:
        variables.append(f"zr{n}{p}")
    for p in phases:
        variables.append(f"xi{n}{p}")
    for p in phases:
        variables.append(f"yi{n}{p}")
    for p in phases:
        variables.append(f"zi{n}{p}")
    for p in phases:
        variables.append(f"v{n}{p}")
    for p in phases:
        variables.append(f"w{n}{p}")
    
end = len(variables)
positions['Mc_Vars'] = (start,end)
#so the order I will do the upper and lower bounds will be zr, zi, v, w, Vr, Vi
inits = loads*18
uppers = jnp.zeros((18,1))
lowers = uppers.copy()
tot_upper = jnp.zeros((inits,1))
tot_lower = tot_upper.copy()
for n in range(1, loads +1):
    uppers = uppers.at[0:6,:].set(1200)
    lowers = lowers.at[0:6,:].set(-1200)
    uppers = uppers.at[6:12,:].set(4160**2)
    lowers = lowers.at[6:12,:].set(0)
    uppers = uppers.at[12:18,:].set(4160)
    lowers = lowers.at[12:18,:].set(-4160)
    tot_upper = tot_upper.at[18*(n-1):18*n,:].set(uppers)
    tot_lower = tot_lower.at[18*(n-1):18*n,:].set(lowers)
print(len(positions))
    
Zline = jnp.array([
 [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
 [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
 [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
])

gen_one = Generator('Node 1 Generator', 1, 12470, 3)
line_one_two = Line('Line 1 to 2', 1, 2, 2000/5280, 3, Zline)
line_one_two = Line('Line 3 to 4', 3, 4, 2500/5280, 3, Zline)
trans_two_three = Transformer('Transformer 2 to 3', 2, 3, 'Y-Y', 12470/4160, 6000000, 12470,0.01+0.06j)
load_4 = Load('Load 4', 4, 1800000, [0.9,0.9,0.9])


both_1 = [line for line in Line.all_lines if line.to_node == 1 and line.from_node == 1]
print(positions['generators'][0])
print(type(powerflow(num_nodes, variables, phases, positions)))
print(positions['Mc_Vars'])
print(init_func(1,phases,vSlack,variables))
MC_mat, b_mat = McC_Load(tot_upper, tot_lower, variables, phases, positions['Mc_Vars'], 4)
print(MC_mat)
A_view = jnp.vstack((init_func(1,phases,vSlack,variables),powerflow(num_nodes, variables, phases, positions)))
np.savetxt("matrix.txt", A_view, fmt="%.3f", delimiter="\t")
np.savetxt("MC_Matrix.txt", MC_mat, fmt="%.3f", delimiter="\t")
print()