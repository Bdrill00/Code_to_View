""" 
Ok, hey there ben.  What we need to do is the following and head straight into it
1) We need to verify that the A and A-hat matrix are correct
    a)Add in code such that the 
"""

from pyomo.environ import *
import pyomo.environ as pyo
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sqrt, atan, Function, lambdify, symbols, Matrix
import math
import math
import jax
import jax.numpy as jnp
from Class_4_Bus import* #Line, Transformer, Generator, Load, powerflow

def states(num_nodes,generators,loads,phases):
    # num_nodes = 4
    # generators = 1
    # loads = 1
    # vSlack = 12470 / jnp.sqrt(3)
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
    # print('Lines',positions['lines'])
    # Generator currents
    start = len(variables)
    for n in range(1, generators+1):
        for p in phases:
            variables.append(f"Igr{p}")
        for p in phases:
            variables.append(f"Igi{p}")
    end = len(variables)
    positions['generators'] = (start, end)
    # print('generators', positions['generators'])
    # Load currents
    # start = len(variables)
    # for n in range(1, loads+1):
    #     for p in phases:
    #         variables.append(f"Ilr{p}")
    #     for p in phases:
    #         variables.append(f"Ili{p}")
    # end = len(variables)
    # positions['loads'] = (start, end)
    # print('Load', positions['loads'])
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
    return variables, num_nodes, positions, tot_upper, tot_lower

def initialization(phases):
    PL = 1800000 #load real power
    angle = math.acos(0.9)
    QL = PL*math.tan(angle)
    # print(QL)
    phs = len(phases)
    vwxyz_vec = jnp.zeros((8*phs,1), dtype=jnp.float64)
    
    InitMag = jnp.array([12470/np.sqrt(3), 12470/np.sqrt(3), 12470/np.sqrt(3), #V1
                        7106.546799,7139.706926,7120.76443,             #V2
                        2247.6, 2269, 2256,           #V3
                        1918, 2061, 1981,             #V4
                        347.9, 323.7, 336.8]).reshape(-1,1)         #I1
                        #1042.8, 970.2, 1009.6]).reshape(-1,1) #I2
    InitAngle = jnp.array([0, -120, 120,
                        -0.3391675422, -120.3439146, 119.6286917,
                        -3.7, -123.5, 116.4,
                        -9.1, -128.3, 110.9,
                        -34.9, -154.2, 85]).reshape(-1,1)
                        # -34.9, -154.2, 85]).reshape(-1,1)

    initReal = jnp.zeros((InitMag.shape[0], 1), dtype=jnp.float64)
    initImag = jnp.zeros((InitMag.shape[0], 1), dtype=jnp.float64)
    for i in range(initReal.shape[0]):
        initReal= initReal.at[i].set(InitMag[i] * jnp.cos(jnp.deg2rad(InitAngle[i])))
        initImag = initImag.at[i].set(InitMag[i] * jnp.sin(jnp.deg2rad(InitAngle[i])))
    v4r = initReal[9:12,:]
    v4i = initImag[9:12,:]
    for i in range(phs):
        zr = (PL*v4r[i]+QL*v4i[i])/(v4r[i]**2+v4i[i]**2)
        xr = zr*v4r[i]**2
        yr = zr*v4i[i]**2
        zi = (PL*v4i[i]-QL*v4r[i])/(v4r[i]**2+v4i[i]**2)
        xi = zi*v4r[i]**2
        yi = zi*v4i[i]**2
        v = v4r[i]**2
        w = v4i[i]**2
        vwxyz_vec = vwxyz_vec.at[i].set(xr)
        vwxyz_vec = vwxyz_vec.at[phs+i].set(yr)
        vwxyz_vec = vwxyz_vec.at[2*phs+i].set(zr)
        vwxyz_vec = vwxyz_vec.at[3*phs+i].set(xi)
        vwxyz_vec = vwxyz_vec.at[4*phs+i].set(yi)
        vwxyz_vec = vwxyz_vec.at[5*phs+i].set(zi)
        vwxyz_vec = vwxyz_vec.at[6*phs+i].set(v)
        vwxyz_vec = vwxyz_vec.at[7*phs+i].set(w)
    result = jnp.zeros(2*len(initReal))
    for i in range(0, len(initReal), len(phases)):
        result = result.at[2*i:2*i+len(phases)].set(initReal[i:i+3].reshape(-1))
        result = result.at[2*i+len(phases):2*i+2*len(phases)].set(initImag[i:i+3].reshape(-1))
        
    return vwxyz_vec,initReal,initImag,result


phases = ['a','b','c']
vwxyz_vec, initReal, initImag, result_vec = initialization(phases)
result_vec = result_vec.reshape(-1,1)
stacked = jnp.vstack([result_vec, vwxyz_vec])
# print(stacked[18:24,:])
# print(vwxyz_vec)


#result is now init conditions where we have v1r, v1i, v2r, v2i,...., Ilr, Ili
variables, num_nodes, positions, tot_upper, tot_lower = states(4,1,1,phases)
# print(positions['Mc_Vars'])
# print(len(positions))
    
Zline = jnp.array([
 [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
 [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
 [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
])

gen_one = Generator('Node 1 Generator', 1, 12470, 3)
line_one_two = Line('Line 1 to 2', 1, 2, 2000/5280, 3, Zline)
line_one_two = Line('Line 3 to 4', 3, 4, 2500/5280, 3, Zline)
nt_ratio = jnp.array(12470, dtype=jnp.float64) / jnp.array(4160, dtype=jnp.float64)
zpu = jnp.array(0.01 + 0.06j, dtype=jnp.complex128)
trans_two_three = Transformer('Transformer 2 to 3', 2, 3, 'Y-Y', nt_ratio, 6000000, 12470,zpu)
load_4 = Load('Load 4', 4, [1800000, 1800000, 1800000], [0.9,0.9,0.9])
# A_matrix = powerflow(num_nodes, variables, phases, positions)
# MC_mat, b_mat = McC_Load(tot_upper, tot_lower, variables, phases, positions['Mc_Vars'], 4)
MC_mat, b_mat = McC_Load(tot_upper, tot_lower, variables, phases, positions['Mc_Vars'], 4)
# print(MC_mat)
init_vs, bees = init_func(1,phases,12470 / jnp.sqrt(3),variables)
A_view = jnp.vstack((init_vs,powerflow(num_nodes, variables, phases, positions)))
b_vee = jnp.vstack([bees,jnp.zeros((A_view.shape[0]-6,1))])
# print(MC_mat.shape, A_view.shape)
np.savetxt("matrix.txt", A_view, fmt="%.3f", delimiter="\t")
np.savetxt("MC_Matrix.txt", MC_mat, fmt="%.3f", delimiter="\t")

model = ConcreteModel()

upper_bounds = stacked +10
lower_bounds = stacked-10

stacked_np = np.array(stacked)
lower_np = np.array(lower_bounds)
upper_np = np.array(upper_bounds)

b = np.array(b_vee)
lenInit = len(stacked_np)
A_np = np.array(A_view)
model.obj = Objective(expr =1)
bhat = np.array(b_mat)
Ahat = np.array(MC_mat)

# Bounds function must accept (model, index)
def bounds_rule(model, i):
    return float(lower_np[i].item()), float(upper_np[i].item())

# Initialize function must accept (model, index)
def init_rule(model, i):
    return float(stacked_np[i].item())

model.n = range(A_np.shape[0])  # or Set(initialize=range(lenInit))

model.x = Var(
    range(lenInit),
    bounds=bounds_rule,
    initialize=init_rule
)
def equality_constraint1(model, i):
    return A_np[i,:]@model.x == b[i].item()
def inequality_constraint1(model, i):
    return Ahat[i,:]@model.x <= bhat[i].item()

model.constraint1 = Constraint(model.n, rule=equality_constraint1)
model.constraint2 = Constraint(model.n, rule=inequality_constraint1)
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True, logfile="baron_prac_data.txt", keepfiles = True)  # 'tee=True' will display solver output in the terminal

model.display()
# print(variables)
# print(positions['loads'])
# print(load_4.imagQ)

