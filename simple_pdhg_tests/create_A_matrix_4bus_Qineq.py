import numpy as np
import pandas as pd
import pyomo.environ as pyo

def get_power_system_matrices():
    # =========================================================================
    # 1. PARAMETERS (Matching your Pyomo model)
    # =========================================================================
    nt = 1.0
    G12, B12 = 10.0, -20.0
    Gx, Bx   = 5.0,  -15.0
    G34, B34 = 8.0,  -18.0
    Vs_i =  0.0
    PL, QL = 0.5, 0.2

    # =========================================================================
    # 2. VARIABLE MAPPING (Index 0 to 15)
    # =========================================================================
    # We define the order of variables in the vector x
    vars_list = [
        'V1r', 'V1i',       # 0, 1
        'V2r', 'V2i',       # 2, 3
        'Vaux_r', 'Vaux_i', # 4, 5
        'V3r', 'V3i',       # 6, 7
        'V4r', 'V4i',       # 8, 9
        'Islack_r', 'Islack_i', # 10, 11
        'Ipx_r', 'Ipx_i',   # 12, 13
        'Isx_r', 'Isx_i',   # 14, 15
        'Vs_r'            # 16
    ]
    
    # Create a dictionary for easy index lookup: {'V1r': 0, ...}
    idx = {name: i for i, name in enumerate(vars_list)}
    
    # Initialize Matrix A (16x16) and Vector b (16)
    A = np.zeros((16, 17))
    b = np.zeros(16)

    # =========================================================================
    # 3. FILL MATRIX A and VECTOR b
    # =========================================================================

    # --- Node 1 Constraints (Source) ---
    # c1_v_real: V1r = Vs_r
    row = 0
    A[row, idx['V1r']] = 1.0
    A[row, idx['Vs_r']] = -1

    # c1_v_imag: V1i = Vs_i
    row = 1
    A[row, idx['V1i']] = 1.0
    b[row] = Vs_i

    # c1_kcl_real: Islack_r + G12(V1r - V2r) - B12(V1i - V2i) = 0
    row = 2
    A[row, idx['Islack_r']] = 1.0
    A[row, idx['V1r']] = G12
    A[row, idx['V2r']] = -G12
    A[row, idx['V1i']] = -B12
    A[row, idx['V2i']] = B12

    # c1_kcl_imag: Islack_i + G12(V1i - V2i) + B12(V1r - V2r) = 0
    row = 3
    A[row, idx['Islack_i']] = 1.0
    A[row, idx['V1i']] = G12
    A[row, idx['V2i']] = -G12
    A[row, idx['V1r']] = B12
    A[row, idx['V2r']] = -B12

    # --- Node 2 Constraints ---
    # c2_real: G12(V2r - V1r) - B12(V2i - V1i) + Ipx_r = 0
    row = 4
    A[row, idx['V2r']] = G12
    A[row, idx['V1r']] = -G12
    A[row, idx['V2i']] = -B12
    A[row, idx['V1i']] = B12
    A[row, idx['Ipx_r']] = 1.0

    # c2_imag: -(G12(V1i - V2i) + B12(V1r - V2r)) + Ipx_i = 0
    # Simplified: G12(V2i - V1i) + B12(V2r - V1r) + Ipx_i = 0
    row = 5
    A[row, idx['V2i']] = G12
    A[row, idx['V1i']] = -G12
    A[row, idx['V2r']] = B12
    A[row, idx['V1r']] = -B12
    A[row, idx['Ipx_i']] = 1.0

    # --- Transformer Currents ---
    # c2_Ipx_real: Ipx_r + nt * Isx_r = 0
    row = 6
    A[row, idx['Ipx_r']] = 1.0
    A[row, idx['Isx_r']] = nt

    # c2_Ipx_imag: Ipx_i + nt * Isx_i = 0
    row = 7
    A[row, idx['Ipx_i']] = 1.0
    A[row, idx['Isx_i']] = nt

    # --- Transformer Voltages (Aux Node) ---
    # c2_Vaux_real: nt * Vaux_r - V2r = 0
    row = 8
    A[row, idx['Vaux_r']] = nt
    A[row, idx['V2r']] = -1.0

    # c2_Vaux_imag: nt * Vaux_i - V2i = 0
    row = 9
    A[row, idx['Vaux_i']] = nt
    A[row, idx['V2i']] = -1.0

    # --- Aux Node KCL ---
    # c2aux_real: Isx_r + Gx(Vaux_r - V3r) - Bx(Vaux_i - V3i) = 0
    row = 10
    A[row, idx['Isx_r']] = 1.0
    A[row, idx['Vaux_r']] = Gx
    A[row, idx['V3r']] = -Gx
    A[row, idx['Vaux_i']] = -Bx
    A[row, idx['V3i']] = Bx

    # c2aux_imag: Isx_i + Gx(Vaux_i - V3i) + Bx(Vaux_r - V3r) = 0
    row = 11
    A[row, idx['Isx_i']] = 1.0
    A[row, idx['Vaux_i']] = Gx
    A[row, idx['V3i']] = -Gx
    A[row, idx['Vaux_r']] = Bx
    A[row, idx['V3r']] = -Bx

    # --- Node 3 Constraints ---
    # c3_real: Gx(V3r - Vaux_r) - Bx(V3i - Vaux_i) + G34(V3r - V4r) - B34(V3i - V4i) = 0
    row = 12
    A[row, idx['V3r']] = Gx + G34
    A[row, idx['Vaux_r']] = -Gx
    A[row, idx['V4r']] = -G34
    A[row, idx['V3i']] = -Bx - B34
    A[row, idx['Vaux_i']] = Bx
    A[row, idx['V4i']] = B34

    # c3_imag: Gx(V3i - Vaux_i) - Bx(V3r - Vaux_r) + G34(V3i - V4i) + B34(V3r - V4r) = 0
    # Note: B terms in Pyomo c3_imag were: -Bx(V3r-Vaux_r) and +B34(V3r-V4r). 
    # Rearranging carefully:
    row = 13
    A[row, idx['V3i']] = Gx + G34
    A[row, idx['Vaux_i']] = -Gx
    A[row, idx['V4i']] = -G34
    A[row, idx['V3r']] = -Bx + B34
    A[row, idx['Vaux_r']] = Bx
    A[row, idx['V4r']] = -B34

    # --- Node 4 Constraints (Linear Load) ---
    # c4_real: G34(V4r - V3r) - B34(V4i - V3i) + PL*V4r - QL*V4i = 0
    row = 14
    A[row, idx['V4r']] = G34 + PL
    A[row, idx['V3r']] = -G34
    A[row, idx['V4i']] = -B34 - QL
    A[row, idx['V3i']] = B34

    # c4_imag: G34(V4i - V3i) + B34(V4r - V3r) + PL*V4i + QL*V4r = 0
    row = 15
    A[row, idx['V4i']] = G34 + PL
    A[row, idx['V3i']] = -G34
    A[row, idx['V4r']] = B34 + QL
    A[row, idx['V3r']] = -B34
    
    return A, b, vars_list

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    A, b, x_names = get_power_system_matrices()

    mm, nn = A.shape
    c = np.zeros(nn)
    c[8] = 1
    c[9] = 1

    # --- Build Pyomo model ---
    m = pyo.ConcreteModel()
    m.N = pyo.RangeSet(0, nn-1)
    m.M = pyo.RangeSet(0, mm-1)

    # Variables x_j >= 0
    m.x = pyo.Var(m.N)

    # Objective: minimize c^T x
    def obj_rule(model):
        return sum(c[j] * model.x[j] for j in model.N)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints A x = b
    def Ax_rule(model, i):
        return sum(A[i, j] * model.x[j] for j in model.N) == b[i]
    m.Ax = pyo.Constraint(m.M, rule=Ax_rule)

    # new constraints
    m.qbup  = pyo.Constraint(expr =       m.x[16] <= 1.1)
    m.qblow = pyo.Constraint(expr = 0.9 <=m.x[16])
    
    # --- Solve ---
    solver = pyo.SolverFactory('ipopt')   # or "gurobi", "cbc", etc.
    results = solver.solve(m, tee=True)
    
    x_vals = [pyo.value(m.x[i]) for i in m.x]
    print("\nSolution Found!")
    print(f"V1: {pyo.value(m.x[0]):.4f} + j{pyo.value(m.x[1]):.4f}")
    print(f"V2: {pyo.value(m.x[2]):.4f} + j{pyo.value(m.x[3]):.4f}")
    print(f"V3: {pyo.value(m.x[6]):.4f} + j{pyo.value(m.x[7]):.4f}")
    print(f"V4: {pyo.value(m.x[8]):.4f} + j{pyo.value(m.x[9]):.4f}")
    print(f"I_slack: {pyo.value(m.x[10]):.4f} + j{pyo.value(m.x[11]):.4f}")
    print(f"Vset: {pyo.value(m.x[16]):.4f}")
    
    print(b)

    