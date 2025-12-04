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
    #create upper and lower bounds
    return variables, num_nodes, positions
def McC_bounds(loads):
    inits = loads*18
    uppers = jnp.zeros((18,1))
    lowers = uppers.copy()
    tot_upper = jnp.zeros((inits,1))
    tot_lower = tot_upper.copy()
    for n in range(1, loads +1):
        # zr, zi, v, w, Vr, Vi
        uppers = uppers.at[0:6,:].set(1200)
        lowers = lowers.at[0:6,:].set(-1200)
        uppers = uppers.at[6:12,:].set(4160**2)
        lowers = lowers.at[6:12,:].set(0)
        uppers = uppers.at[12:18,:].set(4160)
        lowers = lowers.at[12:18,:].set(-4160)
        tot_upper = tot_upper.at[18*(n-1):18*n,:].set(uppers)
        tot_lower = tot_lower.at[18*(n-1):18*n,:].set(lowers)
    return tot_upper, tot_lower
def upper_lower(num_vars,pos):
    #v1r,v1i,v2r,v2i,....,v4r,v4i, Islackr, Islacki, xr, yr, zr, xi, yi, zi, v, w
    Vupperb = 12470/jnp.sqrt(3)
    Vupperb2 = 4160/jnp.sqrt(3)
    Iupperb = 1200
    upper_bs = jnp.zeros(num_vars)
    lower_bs = upper_bs.copy()
    upper_bs = upper_bs.at[pos['lines'][0]:12].set(jnp.ones(12)*(Vupperb))
    lower_bs = lower_bs.at[pos['lines'][0]:12].set(jnp.ones(12)*(-Vupperb))
    upper_bs = upper_bs.at[12:pos['lines'][1]].set(jnp.ones(12)*(Vupperb2))
    lower_bs = lower_bs.at[12:pos['lines'][1]].set(jnp.ones(12)*(-Vupperb2))
    upper_bs = upper_bs.at[pos['generators'][0]:pos['generators'][1]].set(jnp.ones(6)*(Iupperb))
    lower_bs = lower_bs.at[pos['generators'][0]:pos['generators'][1]].set(jnp.ones(6)*(-Iupperb))
    upper_bs = upper_bs.at[pos['Mc_Vars'][0]:pos['Mc_Vars'][0]+6].set(jnp.ones(6)*(Iupperb*Vupperb*Vupperb))
    lower_bs = lower_bs.at[pos['Mc_Vars'][0]:pos['Mc_Vars'][0]+6].set(-jnp.ones(6)*(Iupperb*Vupperb*Vupperb))
    upper_bs = upper_bs.at[pos['Mc_Vars'][0]+6:pos['Mc_Vars'][0]+9].set(jnp.ones(3)*(Iupperb))
    lower_bs = lower_bs.at[pos['Mc_Vars'][0]+6:pos['Mc_Vars'][0]+9].set(jnp.ones(3)*(-Iupperb))
    upper_bs = upper_bs.at[pos['Mc_Vars'][0]+9:pos['Mc_Vars'][0]+15].set(jnp.ones(6)*(Iupperb*Vupperb2*Vupperb2))
    lower_bs = lower_bs.at[pos['Mc_Vars'][0]+9:pos['Mc_Vars'][0]+15].set(-jnp.ones(6)*(Iupperb*Vupperb2*Vupperb2))
    upper_bs = upper_bs.at[pos['Mc_Vars'][0]+15:pos['Mc_Vars'][0]+18].set(jnp.ones(3)*(Iupperb))
    lower_bs = lower_bs.at[pos['Mc_Vars'][0]+15:pos['Mc_Vars'][0]+18].set(jnp.ones(3)*(-Iupperb))
    upper_bs = upper_bs.at[pos['Mc_Vars'][0]+18:pos['Mc_Vars'][1]].set(jnp.ones(6)*(Vupperb2*Vupperb2))
    return upper_bs, lower_bs
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
    result = result.reshape(-1,1)
    stacked_vec = jnp.vstack([result, vwxyz_vec])
    return stacked_vec
def pdhg_step(ii, step):
    c, x_var, lam, mew, Lag, eta1, eta2, eta3, A_mat, b, Ahat, bhat, stuff_change,low_b,high = step
    # eta1 = 0.1*eta1
    # eta2 = 0.1*eta2
    # eta3 = 0.1*eta3
    low_ = low_b.reshape(-1,1)
    high_ = high.reshape(-1,1)
    
    x_var_new = x_var - eta1*(c+A_mat.T@lam+Ahat.T@mew)
    x_var_new = jnp.maximum(low_,x_var_new)
    x_var_new = jnp.minimum(x_var_new, high_)
    
    lam_new = lam + eta2*(A_mat@(2*x_var_new-x_var)-b)
    lam_dif = jnp.linalg.norm(lam_new-lam)
    
    mew_new = mew +eta3*(Ahat@(2*x_var_new-x_var)-bhat)
    mew_new = jnp.maximum(0.0, mew_new)
    mew_dif = jnp.linalg.norm(mew_new-mew)
    lagval = c.T@x_var_new+lam_new.T@(A_mat@x_var_new - b)+mew_new.T@(Ahat@x_var_new-bhat)
    Lag = Lag.at[ii].set(lagval.item())
    
    stuff_change = stuff_change.at[ii,0].set((c.T@x_var_new).item())
    stuff_change = stuff_change.at[ii,1].set(lam_dif.item())
    stuff_change = stuff_change.at[ii,3].set(mew_dif.item())
    
    return c,x_var_new, lam_new, mew_new, Lag, eta1, eta2, eta3, A_mat, b, Ahat, bhat, stuff_change, low_b,high
    # x_var = x_var-eta*()
def pdhg_fun(p_mat,var_x,A_equal,b_equal,A_inequal,b_inequal,inv_eta, low_bs,high_bs, nsteps=1000):
    counter = 1
    pos = p_mat[0]+7
    e_zra = jnp.zeros((A_equal.shape[1],1)).at[pos].set(1)
    # lambda_init = jnp.ones((A_equal.shape[0],1))*1e-2
    # mew_init = jnp.ones((A_inequal.shape[0],1))*1e-2
    #Warm Start 
    lambda_init = jnp.array([[x] for x in [0.7418696412065505, 2.5639730370040305e-05, -0.15608855505888244,1.2017576213985589e-09, 1.2417882364463796e-05, 0.03459346944372994,
                                           -1.5677600876891891e-12, 1.3588207076625198e-12, 4.2971273912978346e-14,-4.9624396025760926e-14, -1.5677600878726407e-12, 8.502475401879074e-13,
                                           0.12455654628535966, 0.04002607319748362, 0.029807422366770805,-0.27815162747179756, -0.11386921884434947, -0.039138757792653196,
                                           0.041552151303434766, 0.013430526090667536, 0.04993162298448178,-0.09279157215356602, -0.038096711428543116, 0.7413921760475987,
                                           0.026458662695608138, -0.0007388173077861157, 1.4598237333888856e-13,2.978785995535494e-13, 0.06347495739362596, 0.9999999999925214,
                                           4.174160322337536e-15, 4.2698642077078037e-11, -4.174799118282323e-15,-2.6512301627654905e-18, -5.68218901947738e-19, 4.7721070087764326e-18]])
    mew_init = jnp.array([[x] for x in [-2.6132605057153377e-10,8.102683472478371e-20,2.8118950109158306e-20,2.087089257738855e-15,
                                        2.0690627542695255e-15,2.0799717748082983e-15,6.626936080224743e-20,-2.9518538682331654e-10,1.6045892517253725e-20,2.0879175676425068e-15,2.0870657398081993e-15,2.079689442251867e-15,
                                        -2.7446874096802613e-18,8.102683472478371e-20,-2.993325968537015e-20,2.086304560416013e-15,2.069062754269593e-15,2.087914162290713e-15,8.116923101866597e-20,-2.9518538682331644e-10,
                                        -1.5725665599542832e-20,-2.613239606747639e-10,2.0870657398081993e-15,2.0876596791262065e-15,-7.422933261618884e-20,-1.142795703032472e-09,6.1014967890916e-20,2.087992607192931e-15,
                                        2.0871349097850287e-15,-3.138017944702033e-08,2.237187498584823e-20,8.102683473500359e-20,-8.237494717898611e-20,-2.613323139498159e-10,-1.4379873310998278e-09,2.084042970501731e-15,
                                        1.8110794393160997e-20,-1.1427957029131857e-09,6.101496789069224e-20,2.086304560416013e-15,2.0871349097850287e-15,-1.5690092852901552e-08,2.2371874985843563e-20,8.102683473500359e-20,
                                        -1.569008658817574e-08,-2.613323166365288e-10,-1.4379873312191137e-09,2.088037419650917e-15,-3.4536144716069716e-16,-6.941336162594784e-17,2.2878880965117225e-16,2.3653197086697195e-16,
                                        2.1686551486819451e-16,-4.750804745361725e-15,2.196588295952866e-16,2.1121019722403964e-16,-3.765620785817874e-05,2.2878880964306897e-16,1.9242869496715241e-16,-8.55257179574691e-16,
                                        -2.03689780683347e-15,1.1472011766362067e-16,2.417135530572613e-16,-6.271825285003417e-07,8.462038665847501e-17,2.2395323953163743e-16]])
    Lag_size = jnp.zeros(nsteps)
    stuff_change = jnp.zeros((nsteps,3))
    eta = 1/(inv_eta+50)
    alpha_1 = eta
    alpha_2 = eta
    alpha_3 = eta
    x_v = var_x.at[pos].set(-1199)
    step_pdhg = (e_zra,x_v,lambda_init,mew_init,Lag_size,alpha_1,alpha_2,alpha_3,A_equal,b_equal,A_inequal,b_inequal,stuff_change,low_bs,high_bs)
    for i in range(nsteps):
        step_pdhg = pdhg_step(i, step_pdhg)
        if i>0:
            lag_list = step_pdhg[4]
            Lag1 = lag_list[i]
            Lag2 = lag_list[i-1]
            if jnp.linalg.norm(Lag1-Lag2) <=1e-3:
                print('termination criteria met')
                break
        counter +=1
    e_zra,x_opt,lambda_opt,mew_opt,Lag_final,alpha_1,alpha_2,alpha_3,\
    A0_dnu,b0_dnu,A1_dnu,b1_dnu,delta_stuff,low_bs,high_bs = step_pdhg

    return Lag_final, x_opt, lambda_opt, mew_opt, delta_stuff, counter
   
   
   
   
   
phases = ['a','b','c']
stacked = initialization(phases)
# for i in range(0,len(stacked)):
#     print(stacked[i])
# result_vec = result_vec.reshape(-1,1)
# stacked = jnp.vstack([result_vec, vwxyz_vec])

#result is now init conditions where we have v1r, v1i, v2r, v2i,...., Ilr, Ili
variables, num_nodes, positions= states(4,1,1,phases)
tot_upper, tot_lower = McC_bounds(1)
Zline = jnp.array([
 [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
 [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
 [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
])

gen_one = Generator('Node 1 Generator', 1, 12470, 3)
line_one_two = Line('Line 1 to 2', 1, 2, 2000/5280, 3, Zline)
line_three_four = Line('Line 3 to 4', 3, 4, 2500/5280, 3, Zline)
nt_ratio = jnp.array(12470, dtype=jnp.float64) / jnp.array(4160, dtype=jnp.float64)
zpu = jnp.array(0.01 + 0.06j, dtype=jnp.complex128)
trans_two_three = Transformer('Transformer 2 to 3', 2, 3, 'Y-Y', nt_ratio, 6000000, 12470,zpu)
load_4 = Load('Load 4', 4, [1800000, 1800000, 1800000], [0.9,0.9,0.9])

MC_mat, b_mc = McC_Load(tot_upper, tot_lower, variables, phases, positions['Mc_Vars'], 4)
init_vs, bees = init_func(1,phases,12470 / jnp.sqrt(3),variables)
A_mat = jnp.vstack((init_vs,powerflow(num_nodes, variables, phases, positions)))
b_vec = jnp.vstack([bees,jnp.zeros((A_mat.shape[0]-6,1))])
K = jnp.vstack([A_mat,MC_mat])
normK = jnp.linalg.norm(K, ord = 2)

all_upper, all_lower = upper_lower(len(variables),positions)
Lag, xv, lambda_, mew, stuff_change, count_ = pdhg_fun(positions['Mc_Vars'],stacked,A_mat,b_vec,MC_mat,b_mc,normK,all_lower,all_upper,nsteps=2000)

obj_change = stuff_change[:,0]
del_lam = stuff_change[:,1]
del_mew = stuff_change[:,2]

x_value = np.ones(100)*np.array(stacked[positions['Mc_Vars'][0]+6])
lowerzr = np.ones(100)


plt.figure(figsize=(8,5))
plt.plot(range(1, len(Lag)+1), np.array(Lag), marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Lagrangian Value')
plt.title('Convergence of Lagrangian')
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True)
plt.tight_layout()
plt.savefig("lagrangian_output.png")  # save figure to file
print("Convergence plot saved as 'lagrangian_output.png'")

plt.figure(figsize=(8,5))
plt.plot(range(0,del_lam.shape[0]), np.array(del_lam), marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Change in Lambda')
plt.title('Convergence of Lambda')
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True)
plt.tight_layout()
plt.savefig("Delta_Lambda.png")  # save figure to file
print("Convergence plot saved as 'Delta_Lambda.png'")

plt.figure(figsize=(8,5))
plt.plot(range(0,del_mew.shape[0]), np.array(del_mew), marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Change in Mew')
plt.title('Convergence of Mew')
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True)
plt.tight_layout()
plt.savefig("Delta_Mew.png")  # save figure to file
print("Convergence plot saved as 'Delta_Mew.png'")

plt.figure(figsize=(8,5))
# plt.plot(range(1, len(x_value)+1), np.array(x_value), marker='o', linestyle='-', label='optimal Zra')
plt.plot(range(1,count_), np.array(obj_change[0:count_-1]), marker='x', linestyle='-', label='tighter Lb')
# plt.plot(range(1,len(lowerzr)+1), lowerzr, marker='x', linestyle='-', label='lb on Zra')
plt.xlabel('Iteration')
plt.ylabel('Value of Zra')
plt.title('X change per iteration')
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.get_major_formatter().set_useOffset(False)
plt.grid(True)
plt.legend()
plt.tight_layout()
print(count_)


import os
# print(Lag)
plt.savefig("better_x.png")  # save figure to file
