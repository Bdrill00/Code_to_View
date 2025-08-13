from pyomo.environ import *
import pyomo.environ as pyo
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sqrt, atan, Function, lambdify, symbols, Matrix
import math
import re
import time

def QuadMcCor(x, y, XU, XL):
    return[
        y >= 2*XU*x - XU**2,
        y >= 2*x*XL - XL**2,
        y <= x*XU - XL*XU + XL*x
    ]
def McCormick(x, y, z, XU, XL, YU, YL):
    return[
        z >= XU*y + x*YU - XU*YU,
        z <= XU*y - XU*YL + x*YL,
        z <= x*YU - XL*YU + XL*y,
        z >= x*YL + XL*y - XL*YL
    ]
def initFunc():
    Vupper = 12470/np.sqrt(3)
    Vlower = 4160/np.sqrt(3)
    Zupper = 1200
    PL = 1800000 #load real power
    QL = (PL/0.9)*math.sin(math.acos(0.9)) #load reactive power
    sV = 3
    nt = 12470/4160
    Vsr = np.array([
        [Vupper * math.cos(math.radians(0))],
        [Vupper * math.cos(math.radians(-120))],
        [Vupper * math.cos(math.radians(120))] 
    ])
    Vsi = np.array([
        [Vupper * math.sin(math.radians(0))],
        [Vupper * math.sin(math.radians(-120))],
        [Vupper * math.sin(math.radians(120))] 
    ])
    ztbase = (4160**2)/6000000 
    ztpu = 0.01+0.06j 
    zt = ztpu*ztbase  
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
    Yline12 = np.linalg.inv(Zline*(2000/5280))
    Yline34 = np.linalg.inv(Zline*(2000/5280))
    Gl12 = Yline12.real
    Bl12 = Yline12.imag
    Gl34 = Yline34.real
    Bl34 = Yline34.imag
    
    upper = np.array([
        Vlower, Vlower, Vlower, Vlower, Vlower, Vlower,
        Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2,
        Zupper, Zupper, Zupper, Zupper, Zupper, Zupper
        ]).reshape(-1,1)
    lower = np.array([
        -Vlower, -Vlower, -Vlower, -Vlower, -Vlower, -Vlower,
        0,0,0,0,0,0,
        -Zupper, -Zupper, -Zupper, -Zupper, -Zupper, -Zupper       
    ]).reshape(-1,1)
    origUpper = upper.copy()
    origLower = lower.copy()
    obj = ['V4r', 'V4i', 'w', 'v', 'z0', 'z1']
    sense0 = ['minimize', 'maximize']
    phase = [0,1,2]
    return PL,QL,sV,Vupper,Vlower,Gl12,Bl12,Gtr,Bti,Gl34,Bl34,Vsr,Vsi,nt,upper,lower,origLower,origUpper,obj,sense0,phase
    
def opf(opfinput):
    start_build = time.time()
    obj, phase,sense0, upper,lower = opfinput
    # obj, sense0, upper, lower =opfinput
    model = ConcreteModel()
    model.n = pyo.RangeSet(0, 2)
    model.V1r = Var(range(sV), bounds=(-9000, 9000), initialize=Vupper)
    model.V1i = Var(range(sV), bounds=(-9000, 9000), initialize=0)
    model.V2r = Var(range(sV), bounds=(-9000, 9000), initialize=Vupper)
    model.V2i = Var(range(sV), bounds=(-9000, 9000), initialize=0)
    model.V3r = Var(range(sV), bounds=(-3000, 3000), initialize=Vlower)
    model.V3i = Var(range(sV), bounds=(-3000, 3000), initialize=0)
    model.V4r = Var(range(sV), bounds=(-3000, 3000))
    model.V4i = Var(range(sV), bounds=(-3000, 3000))
    model.Islackr = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.Islacki = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.Ixr = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.Ixi = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.I2xr = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.I2xi = Var(range(sV), bounds=(-2000, 2000), initialize=0)
    model.z0 = Var(range(sV), bounds=(-1100, 1100))
    model.z1 = Var(range(sV), bounds=(-1100, 1100))
    model.x0 = Var(range(sV), bounds=(-8.7e9, 8.7e9)) #bounds = max(I12*(V4r)**2)
    model.x1 = Var(range(sV), bounds=(-8.7e9, 8.7e9))
    model.y0 = Var(range(sV), bounds=(-8.7e9, 8.7e9))
    model.y1 = Var(range(sV), bounds=(-8.7e9, 8.7e9))
    model.w = Var(range(sV), bounds=(0, 5800000))
    model.v = Var(range(sV), bounds=(0, 5800000))
    model.obj = Objective(expr = getattr(model, obj)[phase], sense=sense0)
    # model.obj = Objective(expr = sum(model.z0[j]+model.z1[j]+model.w[j]+model.v[j]+model.V4r[j]+model.V4i[j] for j in range(sV)), sense= sense0)
    def equality_constraint1(model, i): #Real and imaginary current at node 1
        return -model.Islackr[i] + \
            sum(Gl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(sV)) - \
                sum(Bl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(sV))==0
    def equality_constraint2(model, i):
        return -model.Islacki[i] + \
            sum(Gl12[i,j]*(model.V1i[j]-model.V2i[j]) for j in range(sV)) + \
                sum(Bl12[i,j]*(model.V1r[j]-model.V2r[j]) for j in range(sV))==0
    def equality_constraint3(model, i): #Real and imaginary voltage equating variable V1 to the source voltage
        return Vsr[i] - model.V1r[i] == 0
    def equality_constraint4(model, i):
        return Vsi[i] - model.V1i[i] == 0
    def equality_constraint5(model, i):  #Real and imaginary current at node 2 with Ixr being the current going into the transformer (coming out of node 2)
        return model.Ixr[i] + \
            sum(Gl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(sV)) - \
                sum(Bl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(sV))==0
    def equality_constraint6(model, i):
        return model.Ixi[i] + \
            sum(Gl12[i,j]*(model.V2i[j]-model.V1i[j]) for j in range(sV)) + \
                sum(Bl12[i,j]*(model.V2r[j]-model.V1r[j]) for j in range(sV))==0
    def equality_constraint7(model, i): #Relating current on primary side of the transformer to the current at the secondary side
        return nt*model.Ixr[i] - model.I2xr[i] == 0
    def equality_constraint8(model, i):
        return nt*model.Ixi[i] - model.I2xi[i] == 0
    def equality_constraint9(model, i):#Real and imaginary current across through the transformer given the transformer has a per phase impedance
        return -model.I2xr[i] + \
            sum(Gtr[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(sV)) - \
                sum(Bti[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(sV))==0
    def equality_constraint10(model, i):
        return -model.I2xi[i] + sum(Gtr[i,j]*((1/nt)*model.V2i[j]-model.V3i[j]) for j in range(sV)) + \
            sum(Bti[i,j]*((1/nt)*model.V2r[j]-model.V3r[j]) for j in range(sV))==0
    def equality_constraint11(model, i): #Real and imaginary current at node 3 
        return sum(Gtr[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(sV)) - \
            sum(Bti[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(sV)) + \
                sum(Gl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(sV)) - \
                    sum(Bl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(sV))==0
    def equality_constraint12(model, i):
        return sum(Gtr[i,j]*(model.V3i[j] - (1/nt)*model.V2i[j]) for j in range(sV)) + \
            sum(Bti[i,j]*(model.V3r[j] - (1/nt)*model.V2r[j]) for j in range(sV)) + \
                sum(Gl34[i,j]*(model.V3i[j]-model.V4i[j]) for j in range(sV)) + \
                    sum(Bl34[i,j]*(model.V3r[j]-model.V4r[j]) for j in range(sV))==0
    def equality_constraint13(model, i):#Real and imaginary current at node 4
        return sum(Gl34[i,j]*(model.V4r[j]-model.V3r[j]) for j in range(sV)) -\
            sum(Bl34[i,j]*(model.V4i[j] - model.V3i[j]) for j in range(sV)) + \
                model.z0[i] ==0
    def equality_constraint14(model, i):
        return sum(Gl34[i,j]*(model.V4i[j]-model.V3i[j]) for j in range(sV)) + \
            sum(Bl34[i,j]*(model.V4r[j] - model.V3r[j]) for j in range(sV)) + \
                model.z1[i] ==0
    def equality_constraint15a(model, i):
        return (PL*model.V4r[i] + QL*model.V4i[i]) \
            == model.x0[i] + model.y0[i] # model.z0[i]*(model.w[i] + model.v[i])
    def equality_constraint15b(model, i):
        return model.x0[i] == model.z0[i]*model.w[i]
    def equality_constraint15c(model, i):
        return model.y0[i] == model.z0[i]*model.v[i]
    def equality_constraint16a(model, i):
        return (PL*model.V4i[i] - QL*model.V4r[i]) \
            == model.z1[i]*(model.w[i] + model.v[i]) #model.x1[i] + model.y1[i]
    def equality_constraint16b(model, i):
        return model.x1[i] == model.z1[i]*model.w[i]
    def equality_constraint16c(model, i):
        return model.y1[i] == model.z1[i]*model.v[i]
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
    model.constraint15a = Constraint(model.n, rule = equality_constraint15a)
    model.constraint15b = Constraint(model.n, rule = equality_constraint15b)
    model.constraint15c = Constraint(model.n, rule = equality_constraint15c)
    model.constraint16a = Constraint(model.n, rule = equality_constraint16a)
    model.constraint16b = Constraint(model.n, rule = equality_constraint16b)
    model.constraint16c = Constraint(model.n, rule = equality_constraint16c)
    Constraint_List0 = ['quadMcConstraint0', 'quadMcConstraint1', 'quadMcConstraint2']
    Constraint_List1 = ['quadMcConstraint3', 'quadMcConstraint4', 'quadMcConstraint5']
    Constraint_List2 = ['ineq_constr0', 'ineq_constr1', 'ineq_constr2'] #Per phase bilinearity constraint of z*V4r^2 = z*w = x0
    Constraint_List3 = ['ineq_constr3', 'ineq_constr4', 'ineq_constr5'] #Per phase bilinearity constraint of z*V4i^2 = z*v = y0
    Constraint_List4 = ['ineq_constr6', 'ineq_constr7', 'ineq_constr8'] #Per phase bilinearity constraint of z*V4i^2 = z*v = x1
    Constraint_List5 = ['ineq_constr9', 'ineq_constr10', 'ineq_constr11'] #Per phase bilinearity constraint of z*V4i^2 = z*v = x1
    for name in Constraint_List0 + Constraint_List1 + Constraint_List2 + Constraint_List3 + Constraint_List4 + Constraint_List5:
        setattr(model, name, ConstraintList())
    for i in range(sV):
        constraints = QuadMcCor(model.V4r[i], model.w[i], XU = upper[0:3,0][i], XL = lower[0:3,0][i])    #quadratic mccormick for V4r i iterating per phase
        for c in constraints:
            getattr(model, Constraint_List0[i]).add(c) 
        constraints = QuadMcCor(model.V4i[i], model.v[i], XU = upper[3:6,0][i], XL = lower[3:6,0][i])    #quadratic mccormick for V4i i iterating per phase
        for c in constraints:
            getattr(model, Constraint_List1[i]).add(c)
        constraints = McCormick(model.z0[i], model.w[i], model.x0[i], XU = upper[12:15,0][i], XL = lower[12:15,0][i], YU = upper[6:9,0][i], YL = lower[6:9,0][i])  #McCormick for z*V4r^2 = z*w =x0
        for c in constraints:
            getattr(model, Constraint_List2[i]).add(c)
        constraints = McCormick(model.z0[i], model.v[i], model.y0[i], XU = upper[12:15,0][i], XL = lower[12:15,0][i], YU = upper[9:12,0][i], YL = lower[9:12,0][i])   #McCormick for z*V4i^2 = z*v= y0
        for c in constraints:
            getattr(model, Constraint_List3[i]).add(c)    
        constraints = McCormick(model.z1[i], model.w[i], model.x1[i], XU = upper[-3:][i], XL = lower[-3:][i], YU = upper[6:9,0][i], YL = lower[6:9,0][i])  #McCormick for z*V4r^2 = z*w =x0
        for c in constraints:
            getattr(model, Constraint_List4[i]).add(c)
        constraints = McCormick(model.z1[i], model.v[i], model.y1[i], XU = upper[-3:][i], XL = lower[-3:][i], YU = upper[9:12,0][i], YL = lower[9:12,0][i])   #McCormick for z*V4i^2 = z*v= y0
        for c in constraints:
            getattr(model, Constraint_List5[i]).add(c)
    end_build = time.time()
    start_solve = time.time()
    solver = SolverFactory('baron')
    results = solver.solve(model, tee=False)
    end_solve = time.time()
    # model.display()
    return{
        "task": obj,
        "objective": model.obj(),
        "start_build": start_build,
        "end_build": end_build,
        "start_solve": start_solve,
        "end_solve": end_solve
    }
    
def makeInput():
    opfinput=[]
    for name in obj:
        for r in phase:
            opfinput.append((name,phase[r],sense0[0],upper,lower))
            opfinput.append((name,phase[r],sense0[1],upper,lower))
    return opfinput
# def makeInput():
#     opfinput=[]
#     opfinput.append(('minimize',sense0[0],upper,lower))
#     opfinput.append(('maximize',sense0[1],upper,lower))
#     return opfinput

def timedif(start, end):
    diff = end-start
    return diff
if __name__ == '__main__':
    (PL,QL,sV,Vupper,Vlower,Gl12,Bl12,Gtr,Bti,Gl34,Bl34,
     Vsr,Vsi,nt,upper,lower,origLower,origUpper,obj,sense0,phase) = initFunc()
    iterTime = []
    task = makeInput()
    count = 0
    for item in task:
        solverRes=[]
        solverRes = opf(item)
        buildMod = timedif(solverRes['start_build'], solverRes['end_build'])
        solveMod = timedif(solverRes['start_solve'], solverRes['end_solve'])
        iterTime.append(buildMod+solveMod)
        print("Build Model Time: ", buildMod, 'seconds')
        print("Solve Model Time: ", solveMod, 'seconds')
        # print(f"Build Time for Iteration {count}: {buildMod:.4f} seconds")
        # print(f"Solve Time for Iteration {count}: {solveMod:.4f} seconds")
        # print(f"Tot Iteration {count}: {iterTime[count]:.4f} seconds")
        count=count+1
        
        print(count)
        
    # opf(task)
    # # for number in range(1000):
    # #     makeInput()
