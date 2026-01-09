import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

# Try to import the user's model, strictly for the __main__ block
try:
    from four_bus_mc_pos_ipopt import return_model
except ImportError:
    # Fallback for demonstration if file is missing
    def return_model():
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1,2], bounds=(0, 10)) # Example bounds
        m.c = pyo.Constraint(expr=m.x[1] + m.x[2] == 5)
        return m

def extract_matrices_with_bounds(model):
    """
    Extracts matrices for the standard form:
        min c^T x
        s.t. Ax = b      (Equality Constraints)
             Gx <= h     (General Inequality Constraints)
             Bx <= u     (Box/Bound Constraints)
    """
    # --- 1. Identify Variables (x vector) ---
    vars_list = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    vars_list.sort(key=lambda v: v.name) 
    
    var_map = {id(v): i for i, v in enumerate(vars_list)}
    n_vars = len(vars_list)
    
    # --- 2. Initialize Containers ---
    # General Constraints
    A_rows, b_vals = [], []
    G_rows, h_vals = [], []
    
    # Box Constraints (B and u)
    B_rows, u_vec_vals = [], []
    
    # --- 3. Build Objective Vector (c) ---
    c = np.zeros(n_vars)
    # (Optional: Add objective parsing logic here if needed)

    # --- 4. Process Box Constraints (Variable Bounds) ---
    # Pyomo bounds are attributes, not Constraint objects.
    # We convert: lb <= x <= ub  -->  x <= ub AND -x <= -lb
    
    for i, var in enumerate(vars_list):
        # 4a. Upper Bound (x_i <= UB)
        if var.has_ub():
            # Create a row with 1.0 at the variable's index
            row = np.zeros(n_vars)
            row[i] = 1.0
            B_rows.append(row)
            u_vec_vals.append(pyo.value(var.ub))
            
        # 4b. Lower Bound (x_i >= LB --> -x_i <= -LB)
        if var.has_lb():
            # Create a row with -1.0 at the variable's index
            row = np.zeros(n_vars)
            row[i] = -1.0
            B_rows.append(row)
            u_vec_vals.append(-pyo.value(var.lb))

    # --- 5. Process General Constraints ---
    for constr in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        repn = generate_standard_repn(constr.body)
        
        row = np.zeros(n_vars)
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            if id(var) in var_map:
                row[var_map[id(var)]] = coef
        
        body_const = repn.constant if repn.constant is not None else 0.0
        
        # Handle Equality (Ax = b)
        if constr.equality:
            rhs = pyo.value(constr.upper)
            A_rows.append(row)
            b_vals.append(rhs - body_const)
            
        # Handle Inequality (Gx <= h)
        else:
            # Body <= UB
            if constr.has_ub():
                G_rows.append(row)
                h_vals.append(pyo.value(constr.upper) - body_const)
            
            # LB <= Body  -->  -Body <= -LB
            if constr.has_lb():
                G_rows.append(-row)
                h_vals.append(body_const - pyo.value(constr.lower))

    # --- 6. Convert to Numpy Arrays ---
    A = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b = np.array(b_vals) if b_vals else np.zeros(0)
    
    G = np.array(G_rows) if G_rows else np.zeros((0, n_vars))
    h = np.array(h_vals) if h_vals else np.zeros(0)
    
    B = np.array(B_rows) if B_rows else np.zeros((0, n_vars))
    u = np.array(u_vec_vals) if u_vec_vals else np.zeros(0)

    return A, b, G, h, B, u, c, vars_list

# --- Run the Extraction ---
if __name__ == "__main__":
    model = return_model()
    
    # Extract matrices
    A, b, G, h, B, u, c, variables = extract_matrices_with_bounds(model)

    #np.savetxt("../matrices/v1.csv", variables, delimiter=",")
    np.savetxt("../matrices/A1.csv", A, delimiter=",")
    np.savetxt("../matrices/b1.csv", b, delimiter=",")
    np.savetxt("../matrices/G1.csv", G, delimiter=",")
    np.savetxt("../matrices/h1.csv", h,   delimiter=",")
    np.savetxt("../matrices/B1.csv", B, delimiter=",")
    np.savetxt("../matrices/u1.csv", u,   delimiter=",")
    np.savetxt("../matrices/c1.csv", c,   delimiter=",")
    print(variables)

    
    print(f"Dimensions:")
    print(f" Variables (x): {len(variables)}")
    print(f" Equality (A):  {A.shape}")
    print(f" General Ineq (G): {G.shape}")
    print(f" Box Ineq (B):  {B.shape}")
    print("-" * 30)
    
    print("Variable Vector Order:")
    print([v.name for v in variables])
    print("-" * 30)

    if len(B) > 0:
        print("\nFirst 2 rows of Box Constraints (B):")
        print(B[:2])
        print(f"<= u[:2]: {u[:2]}")
        print("(Note: 1.0 = upper bound check, -1.0 = lower bound check)")