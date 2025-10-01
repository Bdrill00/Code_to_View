from pyomo.environ import *
from multiprocessing import Pool
import numpy as np

model = ConcreteModel()

laborUpper = 20
laborLower = 0
machineUpper = 20
machineLower = 0

upper = np.array([[laborUpper], [machineUpper]])
lower = np.array([[laborLower], [machineLower]])


cVar = [5,3]
index = len(cVar)

model.labor = Var(bounds = (0, 20))
model.machine = Var(bounds = (0,20))
model.z = Var(bounds = (0, 400))

model.obj = Objective(expr = 1) #expr = cVar[0]*model.labor + cVar[1]*model.machine, sense = minimize)
model.constraint0 = Constraint(rule=lambda model: model.z >= 20)
model.constraint1 = Constraint(rule=lambda model: model.labor + model.machine <= 20)
model.constraint2 = Constraint(rule=lambda model: model.z == model.labor * model.machine)

solver = SolverFactory('baron')
results = solver.solve(model, tee = True)
Feas_point = [model.labor(), model.machine(), model.z()]
model.display()
print(Feas_point)
##############################################################################
def solve_model(task):
    
    obj_var, laborU, laborL, machineU, machineL, sense, name, TorF, lilfunc= task
    
    model = ConcreteModel()
    model.labor = Var(bounds=(0, 20))
    model.machine = Var(bounds=(0, 20))
    model.z = Var(bounds = (0, 400))
        
    model.obj = Objective(expr = obj_var(model), sense = sense)
    model.constraintsList = ConstraintList()

    def constr0(model):
        return cVar[0]*model.labor + cVar[1]*model.machine <= cVar[0]*Feas_point[0] + cVar[1]*Feas_point[1]

    def constr1(model):
        return model.z >= 20

    def constr2(model):
        return model.labor + model.machine <= 20

    def constr3(model):
        return laborL <= model.labor
    def constr4(model):
        return model.labor <= laborU

    def constr5(model):
        return machineL <= model.machine
    def constr6(model):
        return model.machine <= machineU

    def McCormick(x, y, z, XU, XL, YU, YL):
        return[
            z >= XU*y + x*YU - XU*YU,
            z <= XU*y - XU*YL + x*YL,
            z <= x*YU - XL*YU + XL*y,
            z >= x*YL + XL*y - XL*YL
        ]

    model.constraint0 = Constraint(rule = constr0)
    model.constraint1 = Constraint(rule = constr1)
    model.constraint2 = Constraint(rule = constr2)
    model.constraint3 = Constraint(rule = constr3)
    model.constraint4 = Constraint(rule = constr4)
    model.constraint5 = Constraint(rule = constr5)
    model.constraint6 = Constraint(rule = constr6)
    
    constraints = McCormick(model.labor, model.machine, model.z, XU = laborU, XL = laborL, YU = machineU, YL = machineL)  #McCormick for z*V4r^2 = z*w =x0
    for c in constraints:
        model.constraintsList.add(c)
        
    # Solve
    solver = SolverFactory('baron')  # or 'ipopt', 'baron', etc.
    result = solver.solve(model, tee=TorF)
    lilfunc
    return {
        "task": name,
        "labor": model.labor(),
        "machine": model.machine(),
        "z": model.z(),
        "objective": model.obj(),
    }
def labor_obj(model):
    return model.labor

def machine_obj(model):
    return model.machine

def dummy():
    return
    
upperIt = np.array([[0], [0]])
lowerIt = np.array([[0], [0]])
tol = 1e-3
count = 0
#obj_var, laborU, laborL, machineU, machineL, sense, name, TorF = task
if __name__ == '__main__':
    
    # for i in range(2):
    while np.linalg.norm(upper - upperIt) > tol or np.linalg.norm(lower-lowerIt) > tol:
        
        task = [
            (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Labor Upper Bound", False, dummy()),
            (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Labor Lower Bound", False, dummy()),
            (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Machine Upper Bound", False, dummy()),
            (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Machine Lower Bound", False, dummy()),
        ]
        
        with Pool(processes = 4) as pool:
            results = pool.map(solve_model, task)
        
        for r in results:
            task = r['task']
            if task == "Labor Upper Bound":
                upperIt[0,0] = upper[0,0]
                upper[0,0] = r['labor']
            elif task == "Labor Lower Bound":
                lowerIt[0,0] = lower[0,0]
                lower[0,0] = r['labor']
            elif task == "Machine Upper Bound":
                upperIt[1,0] = upper[1,0]
                upper[1,0] = r['machine']
            elif task == "Machine Lower Bound":
                lowerIt[1,0] = lower[1,0]
                lower[1,0] = r['machine']
        count = count + 1
    
    task = [
            (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Labor Upper Bound", True, model.display()),
            (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Labor Lower Bound", True, model.display()),
            (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Machine Upper Bound", True, model.display()),
            (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Machine Lower Bound", True, model.display()),
        ]
    with Pool(processes = 4) as pool:
            results = pool.map(solve_model, task)
    for r in results:
            task = r['task']
            if task == "Labor Upper Bound":
                print("Labor Upper Bound is ", r['labor'])
                laborUp = r['labor'] 
            elif task == "Labor Lower Bound":
                print("Labor Lower Bound is ", r['labor']) 
                laborLo = r['labor']
            elif task == "Machine Upper Bound":
                print("Machine Upper Bound is ", r['machine']) 
                machineUp = r['machine']
            elif task == "Machine Lower Bound":
                print("Machine Lower Bound is ", r['machine'])
                machineLo = r['machine']
    
    print(count)

#To get solution with OBBT bounds
model = ConcreteModel()
index = len(cVar)

model.labor = Var(bounds = (0, 20))
model.machine = Var(bounds = (0,20))
model.z = Var(bounds = (0, 400))

model.obj = Objective(expr = cVar[0]*model.labor + cVar[1]*model.machine, sense = minimize)

def constr1(model):
        return model.z >= 20
def constr2(model):
        return model.labor + model.machine <= 20

def constr3(model):
        return laborLo <= model.labor
def constr4(model):
        return model.labor <= laborUp

def constr5(model):
        return machineLo <= model.machine
def constr6(model):
        return model.machine <= machineUp

def McCormick(x, y, z, XU, XL, YU, YL):
        return[
            z >= XU*y + x*YU - XU*YU,
            z <= XU*y - XU*YL + x*YL,
            z <= x*YU - XL*YU + XL*y,
            z >= x*YL + XL*y - XL*YL
        ]

model.constraint1 = Constraint(rule = constr1)
model.constraint2 = Constraint(rule = constr2)
model.constraint3 = Constraint(rule = constr3)
model.constraint4 = Constraint(rule = constr4)
model.constraint5 = Constraint(rule = constr5)
model.constraint6 = Constraint(rule = constr6)

model.constraintsList = ConstraintList()
constraints = McCormick(model.labor, model.machine, model.z, XU = laborUp, XL = laborLo, YU = machineUp, YL = machineLo)  #McCormick for z*V4r^2 = z*w =x0
for c in constraints:
    model.constraintsList.add(c)
        
solver = SolverFactory('baron')
results = solver.solve(model, tee = True)
Feas_point = [model.labor(), model.machine(), model.z()]
model.display()
print(Feas_point)
print(upper)
print(lower)


# #To get solution with original bounds
# model = ConcreteModel()
# index = len(cVar)

# model.labor = Var(bounds = (0, 20))
# model.machine = Var(bounds = (0,20))
# model.z = Var(bounds = (0, 400))

# model.obj = Objective(expr = cVar[0]*model.labor + cVar[1]*model.machine, sense = minimize)

# def constr1(model):
#         return model.z >= 20
# def constr2(model):
#         return model.labor + model.machine <= 20

# def constr3(model):
#         return 0 <= model.labor
# def constr4(model):
#         return model.labor <= 20

# def constr5(model):
#         return 0 <= model.machine
# def constr6(model):
#         return model.machine <= 20

# def McCormick(x, y, z, XU, XL, YU, YL):
#         return[
#             z >= XU*y + x*YU - XU*YU,
#             z <= XU*y - XU*YL + x*YL,
#             z <= x*YU - XL*YU + XL*y,
#             z >= x*YL + XL*y - XL*YL
#         ]

# model.constraint1 = Constraint(rule = constr1)
# model.constraint2 = Constraint(rule = constr2)
# model.constraint3 = Constraint(rule = constr3)
# model.constraint4 = Constraint(rule = constr4)
# model.constraint5 = Constraint(rule = constr5)
# model.constraint6 = Constraint(rule = constr6)

# model.constraintsList = ConstraintList()
# constraints = McCormick(model.labor, model.machine, model.z, XU = 20, XL = 0, YU = 20, YL = 0)  #McCormick for z*V4r^2 = z*w =x0
# for c in constraints:
#     model.constraintsList.add(c)
        
# solver = SolverFactory('baron')
# results = solver.solve(model, tee = True)
# Feas_point = [model.labor(), model.machine(), model.z()]
# model.display()
# print(Feas_point)