""" 
This document is to test the ability of baron to solve a binary integer problem
I want to see how baron can handle a simple binary integer program as it is not handling a more complicated MINLP

"""
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

aj = [6000000, 7000000, 8000000, 9000000, 10000000]
sizeSj = len(aj)

model.x = Var(initialize = 0)
model.y = Var(initialize = 0)
model.z = Var(initialize = 0)
model.sj = Var(range(sizeSj), within = pyo.Binary, initialize = 0)


model.obj = Objective(expr = sum(aj[j]*model.sj[j] for j in range(sizeSj)))

def constraint1(model):
    return model.x**2 + model.y**2 + model.z**2 <= sum(aj[j]*model.sj[j] for j in range(sizeSj))

def constraint2(model):
    return model.x + model.y + model.z >= 1000
    
def constraint3(model):
    return model.z >= 100

def constraint4(model):
    return model.y >= 100

def constraint5(model):
    return model.x >= 100


def constraint7(model):
    return sum(model.sj[j] for j in range(sizeSj)) == 1

model.constraint1 = Constraint(rule = constraint1)
model.constraint2 = Constraint(rule = constraint2)
model.constraint3 = Constraint(rule = constraint3)
model.constraint4 = Constraint(rule = constraint4)
model.constraint5 = Constraint(rule = constraint5)
# model.constraint6 = Constraint(rule = constraint6)
model.constraint7 = Constraint(rule = constraint7)

#choosing the solver and some settings to adjust how long the code will run for
solver = SolverFactory('baron')
# solver.options['MaxIter'] = 1000
solver.options['PrLevel'] = 5
solver.options['MaxTime'] = -1


#Output results from solver as a file 
result = solver.solve(model, tee=True, keepfiles = True)  # 'tee=True' will display solver output in the terminal

model.display()