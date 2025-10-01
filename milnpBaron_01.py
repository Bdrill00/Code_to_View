from pyomo.environ import *
import pyomo.environ as pyo
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sqrt, atan, Function, lambdify, symbols, Matrix
import math
import re
# Create a Pyomo model
model = ConcreteModel()

PL = 1800000
QL = (PL/0.9)*math.sin(math.acos(0.9))
print(QL)
St = 6000000
VoltageH = 12470/np.sqrt(3)
VoltageL = 4160
kVAt = 6000000
nt = 12470/4160

v1ra = VoltageH * math.cos(math.radians(0))
v1rb = VoltageH * math.cos(math.radians(-120))
v1rc = VoltageH * math.cos(math.radians(120))

v1ia = VoltageH * math.sin(math.radians(0))
v1ib = VoltageH * math.sin(math.radians(-120))
v1ic = VoltageH * math.sin(math.radians(120))
Vsr = np.array([
    [v1ra],
    [v1rb],
    [v1rc] 
])
Vsi = np.array([
    [v1ia],
    [v1ib],
    [v1ic]
])
InitI = np.ones((3,1))
Xinit = np.vstack((Vsr, Vsi, Vsr, Vsi, (1/nt)*Vsr, (1/nt)*Vsi, (1/nt)*Vsr, (1/nt)*Vsi, InitI, InitI, InitI, InitI, InitI, InitI))
#Setting up the equations for transformer admittance matrix
ztlow = (VoltageL**2)/kVAt
ztpu = 0.01+0.06j
zt = ztpu*ztlow  
zphase = np.array([ 
    [zt, 0, 0],
    [0, zt, 0],
    [0, 0, zt]
])
Yt = np.linalg.inv(zphase)

Gtr = Yt.real
Bti = Yt.imag 

Zline = np.array([
 [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
 [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
 [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
])

lineOne = 2000/5280
lineTwo = 2500/5280

Zline12 = Zline*lineOne
R12 = Zline12.real
X12 = Zline12.imag
Zline34 = Zline*lineTwo

Yline12 = np.linalg.inv(Zline12)
Yline34 = np.linalg.inv(Zline34)

Gl12 = Yline12.real
Bl12 = Yline12.imag

Gl34 = Yline34.real
Bl34 = Yline34.imag

# 3-phase vector
n=3

#VrIr + ViIi + VrIi + ViIr
VU = [7500, 7500, 7500]
VL = [-7500, -7500, -7500]
IU = [500, 500, 500]
IL = [-500, -500, -500]


xmc = 4
model.n = pyo.RangeSet(0, n-1)

model.V1r = Var(range(n), bounds=(-20000, 20000), initialize=12470)
model.V1i = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.V2r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V2i = Var(range(n), bounds=(-20000, 20000), initialize = 0)
model.V3r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V3i = Var(range(n), bounds=(-20000, 20000), initialize = 0)
model.V4r = Var(range(n), bounds=(-20000, 20000), initialize=12000)
model.V4i = Var(range(n), bounds=(-20000, 20000), initialize = 1)
model.Islackr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Islacki = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Ixr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.Ixi = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.I2xr = Var(range(n), bounds=(-20000, 20000), initialize=0)
model.I2xi = Var(range(n), bounds=(-20000, 20000), initialize=0)

model.z0 = Var(range(n), bounds = (-1500, 1500), initialize = 1)
model.y0 = Var(range(n), bounds = (-10000000000000, 10000000000000), initialize = 0)
model.x0 = Var(range(n), bounds = (-10000000000000, 10000000000000), initialize = 0)
model.z1 = Var(range(n), bounds = (-1500, 1500), initialize = 1)
model.y1 = Var(range(n), bounds = (-10000000000, 10000000000), initialize = 0)
model.x1 = Var(range(n), bounds = (-10000000000, 10000000000), initialize = 0)
model.w = Var(range(n), bounds = (0, 600000000000), initialize = 0)
model.v = Var(range(n), bounds = (0, 600000000000), initialize = 0)

model.XMc0 = Var(range(xmc), bounds = (-200000000, 200000000), initialize = 0)
model.XMc1 = Var(range(xmc), bounds = (-200000000, 200000000), initialize = 0)
model.XMc2 = Var(range(xmc), bounds = (-200000000, 200000000), initialize = 0)
model.XMc3 = Var(range(xmc), bounds = (-200000000, 200000000), initialize = 0)

aj = [6000000, 7000000, 8000000, 9000000, 10000000]
bj = [1, 2, 3, 4, 5]
sizeSj = len(aj)
model.sj = Var(range(sizeSj), within = pyo.Binary)


# Define objective function
model.obj = Objective(expr = sum((bj[j]*model.sj[j] for j in range(sizeSj))))
# model.obj = Objective(expr = sum((bj[j])*model.sj[j] for j in range(sizeSj))) #when this was constraint, it wouldn't stop iterating

# model.obj = Objective(expr = bj[0]*model.sj[0] + bj[1]*model.sj[1] + bj[2]*model.sj[2] + bj[3]*model.sj[3] + bj[4]*model.sj[4] + bj[5]*model.sj[5])
# model.obj = Objective(expr = bj[0]*model.sj[0] + bj[1]*model.sj[1]+bj[2]*model.sj[2])
def equality_constraint1(model, i):
    return -model.Islackr[i] + sum(Gl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n))==0
def equality_constraint2(model, i):
    return -model.Islacki[i] + sum(Gl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n))==0

def equality_constraint3(model, i):
    return Vsr[i] - model.V1r[i] == 0
def equality_constraint4(model, i):
    return Vsi[i] - model.V1i[i] == 0

def equality_constraint5(model, i):
    return model.Ixr[i]+sum(Gl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n))==0
def equality_constraint6(model, i):
    return model.Ixi[i]+sum(Gl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n))==0

def equality_constraint7(model, i):
    return nt*model.Ixr[i] - model.I2xr[i] == 0
def equality_constraint8(model, i):
    return nt*model.Ixi[i] - model.I2xi[i] == 0

def equality_constraint9(model, i):
    return -model.I2xr[i] + sum(Gtr[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n)) - sum(Bti[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n))==0
def equality_constraint10(model, i):
    return -model.I2xi[i] + sum(Gtr[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n)) + sum(Bti[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n))==0

def equality_constraint11(model, i):
    return sum(Gtr[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) - sum(Bti[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n)) - sum(Bl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n))==0
        
def equality_constraint12(model, i):
    return sum(Gtr[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + sum(Bti[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n))==0

def equality_constraint13(model, i):
    return sum(Gl34[i,j]*(model.V4r[j]-model.V3r[j]) for j in range(n)) -sum(Bl34[i,j]*(model.V4i[j] - model.V3i[j]) for j in range(n)) + \
        model.z0[i]==0 
def equality_constraint13a(model, i):
    return (PL*model.V4r[i] + QL*model.V4i[i])/(model.V4r[i]**2 + model.V4i[i]**2)  == model.z0[i]
# def equality_constraint13b(model, i):
#     return model.x0[i] == model.z0[i]*model.w[i] #do mccormick
# def equality_constraint13c(model, i):
#     return model.y0[i] == model.z0[i]*model.v[i] #do mccormick
# def equality_constraint13d(model, i):
#     return model.w[i] == model.V4r[i]**2
# def equality_constraint13e(model, i):
#     return model.v[i] == model.V4i[i]**2

def equality_constraint14(model, i):
    return sum(Gl34[i,j]*(model.V4i[j]-model.V3i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V4r[j] - model.V3r[j]) for j in range(n)) + \
        model.z1[i] == 0 
def equality_constraint14a(model, i):
    return (PL*model.V4i[i] - QL*model.V4r[i])/(model.V4r[i]**2 + model.V4i[i]**2)  == model.z1[i]
# def equality_constraint14b(model, i):
#     return model.x1[i] == model.z1[i]*model.w[i]
# def equality_constraint14c(model, i):
#     return model.y1[i] == model.z1[i]*model.v[i]
        
#make sure we only select one transformer
def equality_constraint15(model):
    return sum(model.sj[j] for j in range(sizeSj)) == 1

def ineq_constr1(model):
    return sum(((aj[j]/3)**2)*model.sj[j] for j in range(sizeSj)) >= model.XMc0[0]**2 + model.XMc0[1]**2 + model.XMc0[2]**2 + model.XMc0[3]**2
def ineq_constr2(model):
    return sum(((aj[j]/3)**2)*model.sj[j] for j in range(sizeSj)) >= model.XMc1[0]**2 + model.XMc1[1]**2 + model.XMc1[1]**2 + model.XMc1[1]**2
def ineq_constr3(model):
    return sum(((aj[j]/3)**2)*model.sj[j] for j in range(sizeSj)) >= model.XMc2[0]**2 + model.XMc2[1]**2 + model.XMc2[2]**2 + model.XMc2[3]**2
#McCormick Envelope Relaxation
def McCormick(x, y, z, XU, XL, YU, YL):
    return[
        z >= XU*y + x*YU - XU*YU,
        z <= XU*y - XU*YL + x*YL,
        z <= x*YU - XL*YU + XL*y,
        z >= x*YL + XL*y - XL*YL
    ]
def QuadMcCor(x, y, XU, XL):
    return[
        y >= 2*XU*x - XU**2,
        y >= 2*x*XL - XL**2,
        y <= x*XU - XL*XU + XL*x
    ]



model.constraint1 = Constraint(model.n, rule=equality_constraint1)
model.constraint2 = Constraint(model.n, rule=equality_constraint2)
model.constraint3 = Constraint(model.n, rule=equality_constraint3)
model.constraint4 = Constraint(model.n, rule=equality_constraint4)
model.constraint5 = Constraint(model.n, rule=equality_constraint5)
model.constraint6 = Constraint(model.n, rule=equality_constraint6)
model.constraint7 = Constraint(model.n, rule=equality_constraint7)
model.constraint8 = Constraint(model.n, rule=equality_constraint8)
model.constraint9 = Constraint(model.n, rule=equality_constraint9)
model.constraint10 = Constraint(model.n, rule=equality_constraint10)
model.constraint11 = Constraint(model.n, rule=equality_constraint11)
model.constraint12 = Constraint(model.n, rule=equality_constraint12)
model.constraint13 = Constraint(model.n, rule=equality_constraint13)
model.constraint13a = Constraint(model.n, rule=equality_constraint13a)
# model.constraint13b = Constraint(model.n, rule=equality_constraint13b)
# model.constraint13c = Constraint(model.n, rule=equality_constraint13c)
# model.constraint13d = Constraint(model.n, rule=equality_constraint13d)
# model.constraint13e = Constraint(model.n, rule=equality_constraint13e)
model.constraint14 = Constraint(model.n, rule=equality_constraint14)
model.constraint14a = Constraint(model.n, rule=equality_constraint14a)
# model.constraint14b = Constraint(model.n, rule=equality_constraint14b)
# model.constraint14c = Constraint(model.n, rule=equality_constraint14c)
model.constraint15 = Constraint(rule=equality_constraint15)

model.ineq_constr1  = pyo.Constraint(rule=ineq_constr1)
model.ineq_constr2  = pyo.Constraint(rule=ineq_constr2)
model.ineq_constr3  = pyo.Constraint(rule=ineq_constr3)

V_u = [580000000, 580000000, 580000000]
V_l = [0,0,0]
W_u = [580000000, 580000000, 580000000]
W_l = [0,0,0]
X_u = [870000000000, 870000000000, 870000000000]
X_l = [-870000000000, -870000000000, -87000000000]
Y_u = [870000000000, 870000000000, 870000000000]
Y_l = [-870000000000, -870000000000, -87000000000]
Z_u = [15000, 15000, 15000]
Z_l = [-15000, -15000, -15000]

VU = [7500, 7500, 7500]
VL = [-7500, -7500, -7500]
IU = [500, 500, 500]
IL = [-500, -500, -500]


XMc_list = ['XMc0', 'XMc1', 'XMc2'] #McCormick variables to replace bilinearities in power inequalities 1, 2, and 3
Constraint_List0 = ['ineq_constr4', 'ineq_constr5', 'ineq_constr6'] #VR*IR constraints for each phase respectively
Constraint_List1 = ['ineq_constr7', 'ineq_constr8', 'ineq_constr9'] #VI*II constraints for each phase respectively
Constraint_List2 = ['ineq_constr10', 'ineq_constr11', 'ineq_constr12'] #VR*II constraints for each phase respectively
Constraint_List3 = ['ineq_constr13', 'ineq_constr14', 'ineq_constr15'] #VI*IR constraints for each phase respectively
Constraint_List4 = ['ineq_constr16', 'ineq_constr17', 'ineq_constr18'] #V4r^2 constraints per phase
Constraint_List5 = ['ineq_constr19', 'ineq_constr20', 'ineq_constr21'] #V4i^2 constraints per phase
Constraint_List6 = ['ineq_constr21', 'ineq_constr22', 'ineq_constr23'] #Per phase bilinearity constraint of z*V4r^2 = z*w = x0
Constraint_List7 = ['ineq_constr24', 'ineq_constr25', 'ineq_constr26'] #Per phase bilinearity constraint of z*V4i^2 = z*v = y0
Constraint_List8 = ['ineq_constr27', 'ineq_constr28', 'ineq_constr29'] #Per phase bilinearity constraint of z*V4i^2 = z*v = x1
Constraint_List9 = ['ineq_constr30', 'ineq_constr31', 'ineq_constr32'] #Per phase bilinearity constraint of z*V4i^2 = z*v = x1
for name in Constraint_List0 + Constraint_List1 + Constraint_List2 + Constraint_List3: #+ Constraint_List4 + Constraint_List5 + Constraint_List6 + Constraint_List7 + Constraint_List8 + Constraint_List9:
    setattr(model, name, ConstraintList())

for i in range(n):
    constraints = McCormick(model.V2r[i], model.Ixr[i], getattr(model, XMc_list[i])[0], XU=VU[i], XL=VL[i], YU=IU[i], YL=IL[i])
    
    for c in constraints:
        getattr(model, Constraint_List0[i]).add(c)
        
    constraints = McCormick(model.V2i[i], model.Ixi[i], getattr(model, XMc_list[i])[1], XU=VU[i], XL=VL[i], YU=IU[i], YL=IL[i])
    
    for c in constraints:
        getattr(model, Constraint_List1[i]).add(c)
        
    constraints = McCormick(model.V2r[i], model.Ixi[i], getattr(model, XMc_list[i])[2], XU=VU[i], XL=VL[i], YU=IU[i], YL=IL[i])
    
    for c in constraints:
        getattr(model, Constraint_List2[i]).add(c)
        
    constraints = McCormick(model.V2i[i], model.Ixr[i], getattr(model, XMc_list[i])[3], XU=VU[i], XL=VL[i], YU=IU[i], YL=IL[i])
    
    for c in constraints:
        getattr(model, Constraint_List3[i]).add(c)
    
    # constraints = QuadMcCor(model.V4r[i], model.w[i], XU = V_u[i], XL = V_l[i])    #quadratic mccormick for V4r i iterating per phase
    # for c in constraints:
    #     getattr(model, Constraint_List4[i]).add(c)
        
    # constraints = QuadMcCor(model.V4i[i], model.v[i], XU = W_u[i], XL = W_l[i])    #quadratic mccormick for V4i i iterating per phase
    # for c in constraints:
    #     getattr(model, Constraint_List5[i]).add(c)
        
    # constraints = McCormick(model.z0[i], model.w[i], model.x0[i], XU = Z_u[i], XL = Z_l[i], YU = W_u[i], YL = W_l[i])  #McCormick for z*V4r^2 = z*w =x0
    # for c in constraints:
    #     getattr(model, Constraint_List6[i]).add(c)
    
    # constraints = McCormick(model.z0[i], model.v[i], model.y0[i], XU = Z_u[i], XL = Z_l[i], YU = V_u[i], YL = V_l[i])   #McCormick for z*V4i^2 = z*v= y0
    # for c in constraints:
    #     getattr(model, Constraint_List7[i]).add(c)
    
    # constraints = McCormick(model.z1[i], model.w[i], model.x1[i], XU = Z_u[i], XL = Z_l[i], YU = W_u[i], YL = W_l[i])  #McCormick for z*V4r^2 = z*w =x0
    # for c in constraints:
    #     getattr(model, Constraint_List8[i]).add(c)
    
    # constraints = McCormick(model.z1[i], model.v[i], model.y1[i], XU = Z_u[i], XL = Z_l[i], YU = V_u[i], YL = V_l[i])   #McCormick for z*V4i^2 = z*v= y0
    # for c in constraints:
    #     getattr(model, Constraint_List9[i]).add(c)
# constraints = McCormick(model, i, 'V2i', 'Ixi', 'XMc1', XU=VU, XL=VL, YU=IU, YL=IL)
#     for c in constraints:
#         model.ineq8to11.add(c)
    
#     constraints = McCormick(model, i, 'V2r', 'Ixi', 'XMc2', XU=VU, XL=VL, YU=IU, YL=IL)
#     for c in constraints:
#         model.ineq12to15.add(c)
    
#     constraints = McCormick(model, i, 'V2i', 'Ixr', 'XMc3', XU=VU, XL=VL, YU=IU, YL=IL)
#     for c in constraints:
#         model.ineq8to11.add(c)
        


# model.ineq_constr1 = Constraint( rule=ineq_constr1)
# model.ineq_constr2 = Constraint(model.n, rule=ineq_constr2)
# model.ineq_constr3 = Constraint(rule=ineq_constr3)
# model.ineq_constr2 = Constraint(rule=ineq_constr2)
solver = SolverFactory('baron')
# solver.options['MaxIter'] = 1000
solver.options['PrLevel'] = 5
solver.options['MaxTime'] = -1
# solver.options['DeltaTerm'] = 1
# solver.options['DeltaT'] = -200
# solver.options['TolRel'] = 1e-6  

result = solver.solve(model, tee=True, logfile="baron_prac_data.txt")  # 'tee=True' will display solver output in the terminal

# Display results
model.display()

V1r_vals = np.array([pyo.value(model.V1r[i]) for i in range(n)]).reshape(-1,1)
V1i_vals = np.array([pyo.value(model.V1i[i]) for i in range(n)]).reshape(-1,1)
V2r_vals = np.array([pyo.value(model.V2r[i]) for i in range(n)]).reshape(-1,1)
V2i_vals = np.array([pyo.value(model.V2i[i]) for i in range(n)]).reshape(-1,1)
V3r_vals = np.array([pyo.value(model.V3r[i]) for i in range(n)]).reshape(-1,1)
V3i_vals = np.array([pyo.value(model.V3i[i]) for i in range(n)]).reshape(-1,1)
V4r_vals = np.array([pyo.value(model.V4r[i]) for i in range(n)]).reshape(-1,1)
V4i_vals = np.array([pyo.value(model.V4i[i]) for i in range(n)]).reshape(-1,1)
Islackr_vals = np.array([pyo.value(model.Islackr[i]) for i in range(n)]).reshape(-1,1)
Islacki_vals = np.array([pyo.value(model.Islacki[i]) for i in range(n)]).reshape(-1,1)
Ixr_vals = np.array([pyo.value(model.Ixr[i]) for i in range(n)]).reshape(-1,1)
Ixi_vals = np.array([pyo.value(model.Ixi[i]) for i in range(n)]).reshape(-1,1)
I2xr_vals = np.array([pyo.value(model.I2xr[i]) for i in range(n)]).reshape(-1,1)
I2xi_vals = np.array([pyo.value(model.I2xi[i]) for i in range(n)]).reshape(-1,1)
# St_vals = np.array([pyo.value(model.St[i]) for i in range(n)]).reshape(-1,1)
# sj_var = np.array([pyo.value(model.sj[i]) for i in range(sizeSj)]).reshape(-1,1)

# Stot_val = np.sum(St_vals)
# print(Stot_val)

#need to switch terms around

# rP1 = sum(Gl12[0,j]*(V2r_vals[j]*(V2r_vals[j] - Vsr[j]) + V2i_vals[j]*(V2i_vals[j] - Vsi[j])) \
#     - Bl12[0,j]*(V2r_vals[j]*(V2i_vals[j] - Vsi[j]) - V2i_vals[j]*(V2r_vals[j] - Vsr[j])) for j in range(n))

# rP2 = sum(Gl12[1,j]*(V2r_vals[j]*(V2r_vals[j] - Vsr[j]) + V2i_vals[j]*(V2i_vals[j] - Vsi[j])) \
#     - Bl12[1,j]*(V2r_vals[j]*(V2i_vals[j] - Vsi[j]) - V2i_vals[j]*(V2r_vals[j] - Vsr[j])) for j in range(n))

# rP3 = sum(Gl12[2,j]*(V2r_vals[j]*(V2r_vals[j] - Vsr[j]) + V2i_vals[j]*(V2i_vals[j] - Vsi[j])) \
#     - Bl12[2,j]*(V2r_vals[j]*(V2i_vals[j] - Vsi[j]) - V2i_vals[j]*(V2r_vals[j] - Vsr[j])) for j in range(n))

# rP = rP1.item() + rP2.item() + rP3.item()

# iQ1 = sum(Gl12[0,j]*(V2r_vals[j]*(V2i_vals[j]-Vsi[j]) - V2i_vals[j]*(V2r_vals[j]-Vsr[j])) \
#       + Bl12[0,j]*(V2r_vals[j]*(V2r_vals[j]-Vsr[j]) - V2i_vals[j]*(V2i_vals[j]-Vsi[j])) for j in range(n))

# iQ2 = sum(Gl12[1,j]*(V2r_vals[j]*(V2i_vals[j]-Vsi[j]) - V2i_vals[j]*(V2r_vals[j]-Vsr[j])) \
#       + Bl12[1,j]*(V2r_vals[j]*(V2r_vals[j]-Vsr[j]) - V2i_vals[j]*(V2i_vals[j]-Vsi[j])) for j in range(n))

# iQ3 = sum(Gl12[2,j]*(V2r_vals[j]*(V2i_vals[j]-Vsi[j]) - V2i_vals[j]*(V2r_vals[j]-Vsr[j])) \
#       + Bl12[2,j]*(V2r_vals[j]*(V2r_vals[j]-Vsr[j]) - V2i_vals[j]*(V2i_vals[j]-Vsi[j])) for j in range(n))

rP1 = sum(Gl12[0,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) + V2i_vals[j]*(Vsi[j] - V2i_vals[j])) \
    - Bl12[0,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) for j in range(n))

rP2 = sum(Gl12[1,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) + V2i_vals[j]*(Vsi[j] - V2i_vals[j])) \
    - Bl12[1,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) for j in range(n))

rP3 = sum(Gl12[2,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) + V2i_vals[j]*(Vsi[j] - V2i_vals[j])) \
    - Bl12[2,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) for j in range(n))

rP = rP1.item() + rP2.item() + rP3.item()
iQ1 = sum(Gl12[0,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) \
      + Bl12[0,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) - V2i_vals[j]*(Vsi[j] - V2i_vals[j])) for j in range(n))

iQ2 = sum(Gl12[1,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) \
      + Bl12[1,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) - V2i_vals[j]*(Vsi[j] - V2i_vals[j])) for j in range(n))

iQ3 = sum(Gl12[2,j]*(V2r_vals[j]*(Vsi[j] - V2i_vals[j]) - V2i_vals[j]*(Vsr[j] - V2r_vals[j])) \
      + Bl12[2,j]*(V2r_vals[j]*(Vsr[j] - V2r_vals[j]) - V2i_vals[j]*(Vsi[j] - V2i_vals[j])) for j in range(n))

iQ = iQ1.item() + iQ2.item() + iQ3.item()
rP11 = rP3.item()
iQ11 = iQ3.item()
something = sqrt(rP11**2 +iQ11**2)
print(something)
# rP01 = rP.item()
# iQ01 = iQ.item()
Smag1 = rP**2+iQ**2
Smag = sqrt(Smag1)
print(Smag)
# print("Transformer selection vector: ", sj_var)



# BinaryS = np.array([pyo.value(model.sj[i]) for i in range(sizeSj)]).reshape(-1,1)


Xn = np.vstack([V1r_vals, V1i_vals, V2r_vals, V2i_vals, V3r_vals, V3i_vals, 
                             V4r_vals, V4i_vals, Islackr_vals, Islacki_vals, 
                             Ixr_vals, Ixi_vals, I2xr_vals, I2xi_vals])

print("Solution Vector:")
names = ["V1", "V2", "V3", "V4", "Islack", "Ix", "I2r" ]
iterate = 0
while iterate <=6:
    iter_while1 = iterate*6
    a =np.sqrt(Xn[iter_while1+0,0]**2 + Xn[iter_while1+3,0]**2)
    b =np.sqrt(Xn[iter_while1+1,0]**2 + Xn[iter_while1+4,0]**2)
    c =np.sqrt(Xn[iter_while1+2,0]**2 + Xn[iter_while1+5,0]**2)
    d = np.degrees(np.arctan2(Xn[iter_while1+3,0], Xn[iter_while1+0,0]))
    e = np.degrees(np.arctan2(Xn[iter_while1+4,0], Xn[iter_while1+1,0]))
    f = np.degrees(np.arctan2(Xn[iter_while1+5,0], Xn[iter_while1+2,0]))
    print(names[iterate])
    print("a-phase magnitude: ", a, "angle: ", d)
    print("b-phase magnitude: ", b, "angle: ", e)
    print("c-phase magnitude: ", c, "angle: ", f)
    iterate +=1
    
Pa = pyomo.environ.value(model.V2r[0]*model.Ixr[0]+model.V2i[0]*model.Ixi[0])
Pb = pyomo.environ.value(model.V2r[1]*model.Ixr[1]+model.V2i[1]*model.Ixi[1])
Pc= pyomo.environ.value(model.V2r[2]*model.Ixr[2]+model.V2i[2]*model.Ixi[2])
Qa = pyomo.environ.value(model.V2i[0]*model.Ixr[0]-model.V2r[0]*model.Ixi[0])
Qb = pyomo.environ.value(model.V2i[1]*model.Ixr[1]-model.V2r[1]*model.Ixi[1])
Qc = pyomo.environ.value(model.V2i[2]*model.Ixr[2]-model.V2r[2]*model.Ixi[2])
Ptot = Pa + Pb + Pc
Qtot = Qa + Qb + Qc

Sa = pyomo.environ.sqrt(Pa**2 + Qa**2)
Sb = pyomo.environ.sqrt(Pb**2 + Qb**2)
Sc = pyomo.environ.sqrt(Pc**2 + Qc**2)
Stot = pyomo.environ.sqrt(Ptot**2 + Qtot**2)

Stot = Sa + Sb + Sc
print("Apparent per phase power at Transformer: ", Sa, "and", Sb, "and", Sc)
print("Apparent power at Transformer: ", Stot)

# with open("ipopt_prac_data.txt", "r") as f:
#     ipoptData = f.read()
    
# pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)\s+([-+]?\d*\.\d+e[+-]?\d+)", re.MULTILINE)


# iterations, objectives, inf_prs, inf_dus = [], [], [], []

# # Search for all matches in the log
# for match in pattern.finditer(ipoptData):
#     iterations.append(int(match.group(1)))        # Iteration number
#     objectives.append(float(match.group(2)))      # Objective value
#     inf_prs.append(float(match.group(3)))         # Primal infeasibility
#     inf_dus.append(float(match.group(4))) 
            
# plt.plot(iterations, inf_prs, marker='o', linestyle='-', label="Primal Infeasibility")
# plt.plot(iterations, inf_dus, marker='s', linestyle='--', label="Dual Infeasibility")
# plt.yscale("log")  # Log scale for better visualization
# plt.xlabel("Iterations")
# plt.ylabel("Error (log scale)")
# plt.title("IPOPT Convergence of IEEE Four")
# plt.legend()
# plt.grid()
# plt.show()

print(range(sizeSj))