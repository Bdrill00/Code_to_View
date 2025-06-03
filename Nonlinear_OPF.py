#In this code, I am rewriting my previous code for readability in an effort to better debug the issues I am having using baron
#As of beginning this code, I am trying to relax the nonlinear constraints of the MINLP I would like to solve using McCormick 
#Currently, full linear relaxation of the nonlinear constraints due to the OPF and transformer selection is evaluated as Infeasible
#This should not be the case and I will delve into that in the proceeding code

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

PL = 1800000 #load real power
QL = (PL/0.9)*math.sin(math.acos(0.9)) #load reactive power
VoltageH = 12470/np.sqrt(3) #primary voltage rms
VoltageL = 4160 #secondary voltage magnitude (need in this form for calculating Zbase for the transformer)
Vlower = 4160/np.sqrt(3)
nt = 12470/4160 #number of turns ratio for transformer
kVAt = 6000000 #rating of the transformer

#Initialize the source voltage  and store as an array
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

#Creating a vector which stores the initial current and voltages I would like to use
InitI = np.ones((3,1))
Xinit = np.vstack((Vsr, Vsi, Vsr, Vsi, (1/nt)*Vsr, (1/nt)*Vsi, (1/nt)*Vsr, (1/nt)*Vsi, InitI, InitI, InitI, InitI, InitI, InitI))

ztbase = (VoltageL**2)/kVAt #this is going to calculate the base 'z' for my transformer
ztpu = 0.01+0.06j #per unit reactance of transformer per phase
zt = ztpu*ztbase  #actual reactance of the lines given per unit measures and z_base

#make it a matrix, find the admittance matrix, and split into real and reactive matrices
zphase = np.array([ 
    [zt, 0, 0],
    [0, zt, 0],
    [0, 0, zt]
])
Yt = np.linalg.inv(zphase)
Gtr = Yt.real
Bti = Yt.imag 

#calculating the line impedance of both lines connected based off of their line lengths
#For 4-bus, these lines are betweens nodes 1-2, and 3-4 with a transformer between them (nodes 2-3)

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

#This is going to be the length of the vector of variables defined after this (n=3 because we have three phases for each variable)
n = 3
model.n = pyo.RangeSet(0, 2)

#Define my pyomo variables for the OPF problem:
model.V1r = Var(range(n), bounds=(-9000, 9000), initialize=12470)
model.V1i = Var(range(n), bounds=(-9000, 9000), initialize=0)
model.V2r = Var(range(n), bounds=(-9000, 9000), initialize=12000)
model.V2i = Var(range(n), bounds=(-9000, 9000), initialize = 0)
model.V3r = Var(range(n), bounds=(-3000, 3000), initialize=Vlower)
model.V3i = Var(range(n), bounds=(-3000, 3000), initialize = 0)
model.V4r = Var(range(n), bounds=(-3000, 3000), initialize=Vlower)
model.V4i = Var(range(n), bounds=(-3000, 3000), initialize = 0)
model.Islackr = Var(range(n), bounds=(-2000, 2000), initialize=0)
model.Islacki = Var(range(n), bounds=(-2000, 2000), initialize=0)
model.Ixr = Var(range(n), bounds=(-2000, 2000), initialize=0)
model.Ixi = Var(range(n), bounds=(-2000, 2000), initialize=0)
model.I2xr = Var(range(n), bounds=(-2000, 2000), initialize=0)
model.I2xi = Var(range(n), bounds=(-2000, 2000), initialize=0)


#This is for our transformer selection portion of the problem:
#aj represents the rating of transformer j with sj being 1 when its selected and 0 when not selected
aj = [6000000, 7000000, 8000000, 9000000, 10000000]
sizeSj = len(aj)
model.sj = Var(range(sizeSj), within = pyo.Binary)

#So the goal of this optimization problem besides testing my knowledge of McCormick linearization of nonlinearities
#is to solve the OPF with an additional transformer selection
#In rewriting this code, the first thing I would like to try is to linearize the OPF with McCormick
#From there I will add in the transformer selection and see what happens

model.obj = Objective(expr = 1)

#For Pyomo, we want to define our functions and then call them to be modeled
#In looking at it now, some of the functions could be removed and iteratively created if we were to desire to do more with this

#Real and imaginary current at node 1
def equality_constraint1(model, i):
    return -model.Islackr[i] + sum(Gl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n))==0
def equality_constraint2(model, i):
    return -model.Islacki[i] + sum(Gl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(n))==0

#Real and imaginary voltage equating variable V1 to the source voltage
def equality_constraint3(model, i):
    return Vsr[i] - model.V1r[i] == 0
def equality_constraint4(model, i):
    return Vsi[i] - model.V1i[i] == 0

#Real and imaginary current at node 2 with Ixr being the current going into the transformer (coming out of node 2)
def equality_constraint5(model, i):
    return model.Ixr[i] + sum(Gl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n)) - sum(Bl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n))==0
def equality_constraint6(model, i):
    return model.Ixi[i] + sum(Gl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(n)) + sum(Bl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(n))==0

#Relating current on primary side of the transformer to the current at the secondary side
def equality_constraint7(model, i):
    return nt*model.Ixr[i] - model.I2xr[i] == 0
def equality_constraint8(model, i):
    return nt*model.Ixi[i] - model.I2xi[i] == 0

#Real and imaginary current across through the transformer given the transformer has a per phase impedance
def equality_constraint9(model, i):
    return -model.I2xr[i] + sum(Gtr[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n)) - sum(Bti[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n))==0
def equality_constraint10(model, i):
    return -model.I2xi[i] + sum(Gtr[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(n)) + sum(Bti[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(n))==0

#Real and imaginary current at node 3 
def equality_constraint11(model, i):
    return sum(Gtr[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) - sum(Bti[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n)) - sum(Bl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n))==0
def equality_constraint12(model, i):
    return sum(Gtr[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(n)) + sum(Bti[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(n)) + \
        sum(Gl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(n))==0

#Real and imaginary current at node 4
#Note that our only nonlinearities exist in here as, with V-I formulation, equations for power are nonlinear
def equality_constraint13(model, i):
    return sum(Gl34[i,j]*(model.V4r[j]-model.V3r[j]) for j in range(n)) -sum(Bl34[i,j]*(model.V4i[j] - model.V3i[j]) for j in range(n)) + \
        (PL*model.V4r[i] + QL*model.V4i[i])/(model.V4r[i]**2 + model.V4i[i]**2) ==0
        
def equality_constraint14(model, i):
    return sum(Gl34[i,j]*(model.V4i[j]-model.V3i[j]) for j in range(n)) + sum(Bl34[i,j]*(model.V4r[j] - model.V3r[j]) for j in range(n)) + \
        (PL*model.V4i[i] - QL*model.V4r[i])/(model.V4r[i]**2 + model.V4i[i]**2)  ==0

#adding the constraints to the model
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
model.constraint14 = Constraint(model.n, rule=equality_constraint14)
    
#choosing the solver and some settings to adjust how long the code will run for
solver = SolverFactory('baron')
# solver.options['MaxIter'] = 1000
solver.options['PrLevel'] = 5
solver.options['MaxTime'] = -1


#Output results from solver as a file 
result = solver.solve(model, tee=True, logfile="baron_prac_data.txt", keepfiles = True)  # 'tee=True' will display solver output in the terminal

model.display()

#Export the variables from the model to be able to use for post processing
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

#Extract the values of current and voltages as magnitude and angle instead of real and imaginary
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


#Notes:
#I have just run the OPF problem.  From here I am going to add in a dummy variable 'z0' and 'z1' to replace the nonlinearities with
#Nonlinearity being (P*V4r + Q*V4i)/(V4r^2 + V4i^2) and (P*V4i - Q*V4r)/(V4r^2 + V4i^2)
#I will experiment which equivalent formulations of z equalling these constraints will baron not like
#This set of equality constraints with the dummy variables worked
# def equality_constraint15(model, i):
#     return model.z0[i] == (PL*model.V4r[i] + QL*model.V4i[i])/(model.V4r[i]**2 + model.V4i[i]**2)
# def equality_constraint16(model, i):
#     return model.z1[i] == (PL*model.V4i[i] - QL*model.V4r[i])/(model.V4r[i]**2 + model.V4i[i]**2)
#This worked as well
# def equality_constraint15(model, i):
#     return model.z0[i]*(model.V4r[i]**2 + model.V4i[i]**2) == (PL*model.V4r[i] + QL*model.V4i[i])
# def equality_constraint16(model, i):
#     return model.z1[i]*(model.V4r[i]**2 + model.V4i[i]**2) == (PL*model.V4i[i] - QL*model.V4r[i])
#Time to try adding in McCormick for this
#I am going to try to linearize the V4r^2 and V4i^2 terms first
#I linearized with the quadratic version of McCormick; with the current version though my problem is infeasible
#I changed the bounds both for my variables and my McCormick equations which allowed me to find a feasible solution
#I want to figure out where this breaks so instead of adding in more constraints I will try to break this problem
#My lower bound was 0 originally which was incorrect.  0 is a lower bound for w and v but not for V4
print(Vlower)