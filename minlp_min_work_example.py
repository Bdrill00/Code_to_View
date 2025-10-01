from pyomo.environ import *
import pyomo.environ as pyo
import numpy as np
from numpy.linalg import solve
from sympy import sqrt, atan, Function, lambdify, symbols, Matrix

# Create a Pyomo model
model = ConcreteModel()
Yrij = 0.6283
Yiij = -1.2878
V1r = 12470
V1i = 0
P = 1800000
Q = 871779.79

model.obj = Objective(expr = 1)
model.V2r = Var(bounds = (-20000, 20000), initialize = 12740)
model.V2i = Var(bounds = (-20000, 20000), initialize = 0)
model.a = Var(bounds = (-1500, 1500), initialize = 0)
model.b = Var(bounds = (-1500, 1500), initialize = 0)

#replacing the commented equations with the uncommented one will change the answer you get despite them representing the same thing.
#the currently commented out equations give me the answer I expect to get
#the currently uncommented equations give me an unexpected answer

# # def equality_constraint0(model):
# #     return Yrij*(V1r-model.V2r) - Yiij*(V1i-model.V2i) - (P*model.V2r + Q*model.V2i)/(model.V2r**2 + model.V2i**2) == 0
# def equality_constraint0(model):
#     return Yrij*(V1r-model.V2r)*(model.V2r**2 + model.V2i**2) - Yiij*(V1i-model.V2i)*(model.V2r**2 + model.V2i**2) == (P*model.V2r + Q*model.V2i)
# # def equality_constraint1(model):
# #     return Yrij*(V1i-model.V2i) + Yiij*(V1r-model.V2r) - (P*model.V2i - Q*model.V2r)/(model.V2r**2 + model.V2i**2) == 0
# def equality_constraint1(model):
#     return Yrij*(V1i-model.V2i)*(model.V2r**2 + model.V2i**2) + Yiij*(V1r-model.V2r)*(model.V2r**2 + model.V2i**2) == (P*model.V2i - Q*model.V2r)

# model.constraint0 = Constraint(rule = equality_constraint0)
# model.constraint1 = Constraint(rule = equality_constraint1)

#For the following, comment lines 26 to 36
#The following constraints work similarly to the prior code:
#The commented out constraints provide a feasible and correct solution (solved analytically and with a newton raphson solver)
#The uncommented out contraints are said to be an infeasible problem despite the congruence to the previous constraints

# def equality_constraint_00(model):
#     return Yrij*(V1r-model.V2r) - Yiij*(V1i-model.V2i)- model.a ==0
# def equality_constraint_01(model):
#     return (P*model.V2r + Q*model.V2i)/(model.V2r**2 + model.V2i**2) == model.a
# def equality_constraint_02(model):
#     return  Yrij*(V1i-model.V2i) + Yiij*(V1r-model.V2r) - model.b == 0
# def equality_constraint_03(model):
#     return (P*model.V2i - Q*model.V2r)/(model.V2r**2 + model.V2i**2) == model.b

def equality_constraint_00(model):
    return Yrij*(V1r-model.V2r) - Yiij*(V1i-model.V2i)- model.a ==0
def equality_constraint_01(model):
    return (P*model.V2r + Q*model.V2i) == model.a*(model.V2r**2 + model.V2i**2)
def equality_constraint_02(model):
    return  Yrij*(V1i-model.V2i) + Yiij*(V1r-model.V2r) - model.b == 0
def equality_constraint_03(model):
    return (P*model.V2i - Q*model.V2r) == model.b*(model.V2r**2 + model.V2i**2)



model.constraint00 = Constraint(rule = equality_constraint_00)
model.constraint01 = Constraint(rule = equality_constraint_01)
model.constraint02 = Constraint(rule = equality_constraint_02)
model.constraint03 = Constraint(rule = equality_constraint_03)
solver = SolverFactory('baron')
solver.options['PrLevel'] = 5
solver.options['MaxTime'] = -1


result = solver.solve(model, tee=True, keepfiles=True, logfile="baron_prac_data.txt")  

# Display results
model.display()