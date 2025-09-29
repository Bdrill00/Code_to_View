from pyomo.environ import *
from multiprocessing import Pool
import numpy as np

upperIt = np.zeros((2, 1))
lowerIt = np.zeros((2, 1))
upperDummy = np.zeros((2,1))
lowerDummy = np.zeros((2,1))
tol = 1e-10
count = 0
upper = np.array([[20.00], [20.00]])
lower = np.array([[0.00], [0.00]])
upper1 = np.array([[20.00], [20.00]])
lower1 = np.array([[0.00], [0.00]])
dumV = 0
eofunc = 0
cVar = [5, 3]
zBound = 80
xplusy = 20
def McCormick(x, y, z, XU, XL, YU, YL):
    return[
        z >= XU*y + x*YU - XU*YU,
        z <= XU*y - XU*YL + x*YL,
        z <= x*YU - XL*YU + XL*y,
        z >= x*YL + XL*y - XL*YL
    ]
    
def feasible_p():
    model = ConcreteModel()
    model.labor = Var(bounds=(lower[0,0], upper[0,0]))
    model.machine = Var(bounds=(lower[1,0], upper[1,0]))
    model.z = Var(bounds=(0, 400))
    model.obj = Objective(expr=0, sense = maximize)  
    model.constraint0 = Constraint(rule=lambda model: model.z == zBound)
    model.constraint1 = Constraint(rule=lambda model: model.labor + model.machine <= xplusy)
    model.constraint2 = Constraint(rule=lambda model: model.z == model.labor * model.machine)
    solver = SolverFactory('baron')
    results = solver.solve(model, tee=True)
    Feas_point = [model.labor(), model.machine(), model.z()]
    model.display()
    return Feas_point

def opt_p():
    model = ConcreteModel()
    model.labor = Var(bounds=(lower1[0,0], upper1[0,0]))
    model.machine = Var(bounds=(lower1[1,0], upper1[1,0]))
    model.z = Var(bounds=(0, zBound))  # Adjust if needed
    model.obj = Objective(expr=cVar[0]*model.labor + cVar[1]*model.machine)  
    model.constraint0 = Constraint(rule=lambda model: model.z == zBound)
    model.constraint1 = Constraint(rule=lambda model: model.labor + model.machine <= xplusy)
    model.constraint2 = Constraint(rule=lambda model: model.z == model.labor * model.machine)
    
    solver = SolverFactory('baron')
    results = solver.solve(model, tee=False)
    opt_point = [model.labor(), model.machine(), model.z(), model.obj()]
    return opt_point

def solve_model(task):   
    laborU, laborL, machineU, machineL, sense, name, TorF, Feas_p = task
    
    model = ConcreteModel()
    model.labor = Var(bounds=(lower[0,0], upper[0,0]), initialize = Feas_p[0])
    model.machine = Var(bounds=(lower[1,0], upper[1,0]), initialize = Feas_p[1])
    model.z = Var(bounds=(0, zBound))  # Adjust if needed
    model.obj = Objective(expr = model.labor + model.machine, sense = sense)
    model.constraintsList = ConstraintList()
    model.constraint0 = Constraint(rule=lambda model: cVar[0]*model.labor + cVar[1]*model.machine <= cVar[0]*Feas_p[0] + cVar[1]*Feas_p[1])
    model.constraint1 = Constraint(rule=lambda model: model.z == zBound)
    model.constraint2 = Constraint(rule=lambda model: model.labor + model.machine <= xplusy)
    model.constraint3 = Constraint(rule=lambda model: model.labor >= laborL)
    model.constraint4 = Constraint(rule=lambda model: model.labor <= laborU) 
    model.constraint5 = Constraint(rule=lambda model: model.machine >= machineL)
    model.constraint6 = Constraint(rule=lambda model: model.machine <= machineU)
    model.constraint7 = Constraint(rule=lambda model: laborU*laborU -2*laborU*model.labor +model.labor*model.labor >= 0)
   
    constraints = McCormick(model.labor, model.machine, model.z, XU = laborU, XL = laborL, YU = machineU, YL = machineL)  #McCormick for z*V4r^2 = z*w =x0
    for c in constraints:
        model.constraintsList.add(c)
    solver = SolverFactory('baron')  # or 'ipopt', 'baron', etc.
    results = solver.solve(model, tee=TorF)
    return {
        "task": name,
        "labor": model.labor(),
        "machine": model.machine(),
        "z": model.z(),
        "objective": model.obj(),
    }
    
def Orig_problem(laborU, laborL, machineU, machineL, name, TorF, displayFunc):
    model = ConcreteModel()
    model.labor = Var(bounds=(lower[0,0], upper[0,0]))
    model.machine = Var(bounds=(lower[1,0], upper[1,0]))
    model.z = Var(bounds=(0, zBound))
    model.obj = Objective(expr = cVar[0]*model.labor + cVar[1]*model.machine, sense = minimize)
    model.constraintsList = ConstraintList()
    model.constraint1 = Constraint(rule=lambda model: model.z == zBound)
    model.constraint2 = Constraint(rule=lambda model: model.labor + model.machine <= xplusy)
    constraints = McCormick(model.labor, model.machine, model.z, XU = laborU, XL = laborL, YU = machineU, YL = machineL)  #McCormick for z*V4r^2 = z*w =x0
    for c in constraints:
        model.constraintsList.add(c)
    solver = SolverFactory('baron')  # or 'ipopt', 'baron', etc.
    results = solver.solve(model, tee=TorF)
    displayFunc
    return {
        "task": name,
        "labor": model.labor(),
        "machine": model.machine(),
        "z": model.z(),
        "objective": model.obj(),      
    }
    
# def make_task(labor_obj, machine_obj, upper, lower, TorF, Fpoint):
#     return [
#         (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Labor Upper Bound", TorF, Fpoint),
#         (labor_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Labor Lower Bound", TorF, Fpoint),
#         (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Machine Upper Bound", TorF, Fpoint),
#         (machine_obj, upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Machine Lower Bound", TorF, Fpoint),   
#     ]
def make_task1(upper, lower, TorF, Fpoint):
    return [
        (upper[0,0], lower[0,0], upper[1,0], lower[1,0], maximize, "Upper Bound", TorF, Fpoint),
        (upper[0,0], lower[0,0], upper[1,0], lower[1,0], minimize, "Lower Bound", TorF, Fpoint),
    ]
def labor_obj(model):
    return model.labor
def machine_obj(model):
    return model.machine
def z_obj(model):
    return model.z
def dummyM():
    return
def displayM(model):
    return model.display()
# def solve_results(results, task, upperDummy, lowerDummy):
#     for r in results:
#             task = r['task']
#             if task == "Labor Upper Bound":
#                 upperDummy[0,0] = r['labor']
#             elif task == "Labor Lower Bound":
#                 lowerDummy[0,0] = r['labor']
#             elif task == "Machine Upper Bound":
#                 upperDummy[1,0] = r['machine']
#             elif task == "Machine Lower Bound":
#                 lowerDummy[1,0] = r['machine']
def solve_results1(results, task, upperDummy, lowerDummy):
    for r in results:
            task = r['task']
            if task == "Upper Bound":
                upperDummy[0,0] = r['labor']
                upperDummy[1,0] = r['machine']
            elif task == "Lower Bound":
                lowerDummy[0,0] = r['labor']
                lowerDummy[1,0] = r['machine']  
                             
# def update_func(results, task, upperIt, upper, lowerIt, lower):
#     for r in results:
#             task = r['task']
#             if task == "Labor Upper Bound":
#                 upperIt[0,0] = upper[0,0]
#                 upper[0,0] = r['labor']
#             elif task == "Labor Lower Bound":
#                 lowerIt[0,0] = lower[0,0]
#                 lower[0,0] = r['labor']
#             elif task == "Machine Upper Bound":
#                 upperIt[1,0] = upper[1,0]
#                 upper[1,0] = r['machine']
#             elif task == "Machine Lower Bound":
#                 lowerIt[1,0] = lower[1,0]
#                 lower[1,0] = r['machine']
def update_func1(results, task, upperIt, upper, lowerIt, lower):
    for r in results:
            task = r['task']
            if task == "Upper Bound":
                upperIt[0,0] = upper[0,0]
                upper[0,0] = r['labor']
                upperIt[1,0] = upper[1,0]
                upper[1,0] = r['machine']
            elif task == "Lower Bound":
                lowerIt[0,0] = lower[0,0]
                lower[0,0] = r['labor']
                lowerIt[1,0] = lower[1,0]
                lower[1,0] = r['machine']
def print_results(taskVal, laborVal, machineVal, zVal, objVal):
    print(taskVal)
    print('Labor: ', laborVal)
    print('Machine: ', machineVal)
    print('Z: ', zVal)
    print('Objective Value: ', objVal)
    
##########################################################################

if __name__ == '__main__':
    Fpoint = feasible_p()
    print(Fpoint)
    for number in range(100):
        task = make_task1(upper, lower, False, Fpoint)
        with Pool(processes = 2) as pool:
            results = pool.map(solve_model, task)
        solve_results1(results, task, upperDummy, lowerDummy)
                
        if np.linalg.norm(upperDummy - upper) < tol and np.linalg.norm(lowerDummy - lower) < tol:
            result1 = Orig_problem(20, 0, 20, 0, "Original Problem with Original Bounds", False, dummyM)
            print_results(result1['task'], result1['labor'], result1['machine'], result1['z'], result1['objective'])            
            print(f"Converged after {count} iterations.")
            # result = Orig_problem(upperDummy[0,0], lowerDummy[0,0], upperDummy[1,0], lowerDummy[1,0], 'Plugging Updated McCormick Bounds into Original Problem', False, displayM)
            # print_results(result['task'], result['labor'], result['machine'], result['z'], result['objective'])
            break
        
        update_func1(results, task, upperIt, upper, lowerIt, lower)
        count = count + 1
    optimalp= opt_p()
    print(optimalp)
    test = np.array([1,2,3]).reshape(1,-1)
    print(test)
    