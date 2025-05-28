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


#replacing the commented equations with the uncommented one will change the answer you get despite them representing the same thing.
#the currently commented out equations give me the answer I expect to get
#the currently uncommented equations give me an unexpected answer

# def equality_constraint0(model):
#     return Yrij*(V1r-model.V2r) - Yiij*(V1i-model.V2i) - (P*model.V2r + Q*model.V2i)/(model.V2r**2 + model.V2i**2) == 0
def equality_constraint0(model):
    return Yrij*(V1r-model.V2r)*(model.V2r**2 + model.V2i**2) - Yiij*(V1i-model.V2i)*(model.V2r**2 + model.V2i**2) == (P*model.V2r + Q*model.V2i)
# def equality_constraint1(model):
#     return Yrij*(V1i-model.V2i) + Yiij*(V1r-model.V2r) - (P*model.V2i - Q*model.V2r)/(model.V2r**2 + model.V2i**2) == 0
def equality_constraint1(model):
    return Yrij*(V1i-model.V2i)*(model.V2r**2 + model.V2i**2) + Yiij*(V1r-model.V2r)*(model.V2r**2 + model.V2i**2) == (P*model.V2i - Q*model.V2r)

model.constraint0 = Constraint(rule = equality_constraint0)
model.constraint1 = Constraint(rule = equality_constraint1)

solver = SolverFactory('baron')
solver.options['PrLevel'] = 5
solver.options['MaxTime'] = -1


result = solver.solve(model, tee=True, keepfiles=True, logfile="baron_prac_data.txt")  

# Display results
model.display()