import pyomo.environ as pyo

def create_power_system_model():
    m = pyo.ConcreteModel()

    # =========================================================================
    # 1. DATA / PARAMETERS (Replace these with your actual values)
    # =========================================================================
    
    # Grid Parameters
    m.nt = pyo.Param(initialize=1.0)  # Turns ratio
    
    # Line 1-2 Admittance
    m.G12 = pyo.Param(initialize=10.0)
    m.B12 = pyo.Param(initialize=-20.0)
    
    # Transformer/Line 2-3 Parameters (Gx, Bx)
    m.Gx = pyo.Param(initialize=5.0)
    m.Bx = pyo.Param(initialize=-15.0)
    
    # Line 3-4 Admittance
    m.G34 = pyo.Param(initialize=8.0)
    m.B34 = pyo.Param(initialize=-18.0)
    
    # Source Voltage (Node 1)
    m.Vs_r = pyo.Param(initialize=1.0) # 1.0 pu
    m.Vs_i = pyo.Param(initialize=0.0) # 0.0 pu
    
    # Load at Node 4
    m.PL = pyo.Param(initialize=0.5) 
    #m.QL = pyo.Param(initialize=0.2)

    # =========================================================================
    # 2. VARIABLES
    # =========================================================================
    
    # Voltages (Real and Imaginary) for Nodes 1, 2, 3, 4
    # Initializing near 1.0 avoids division by zero in the Node 4 constraints
    m.V1r = pyo.Var(initialize=1.0)
    m.V1i = pyo.Var(initialize=0.0)
    
    m.V2r = pyo.Var(initialize=1.0)
    m.V2i = pyo.Var(initialize=0.0)
    
    m.V3r = pyo.Var(initialize=1.0)
    m.V3i = pyo.Var(initialize=0.0)
    
    m.V4r = pyo.Var(initialize=1.0)
    m.V4i = pyo.Var(initialize=0.0)
    
    # Slack Currents
    m.Islack_r = pyo.Var(initialize=0.0)
    m.Islack_i = pyo.Var(initialize=0.0)

    # Transformer primary current
    m.Ipx_r = pyo.Var(initialize=0.0)
    m.Ipx_i = pyo.Var(initialize=0.0)

    # Transformer secondary current
    m.Isx_r = pyo.Var(initialize=0.0)
    m.Isx_i = pyo.Var(initialize=0.0)

    # Transformer aux voltages (if needed)
    m.Vaux_r = pyo.Var(initialize=0.0)
    m.Vaux_i = pyo.Var(initialize=0.0)

    m.QL = pyo.Var(initialize=0.2, bounds=(-0.9, 0.9))

    # =========================================================================
    # 3. CONSTRAINTS
    # =========================================================================

    # --- Node 1 Constraints ---
    
    # Voltage Source definition
    m.c1_v_real = pyo.Constraint(expr= m.Vs_r - m.V1r == 0)
    m.c1_v_imag = pyo.Constraint(expr= m.Vs_i - m.V1i == 0)
    
    # KCL at Node 1 (Real)
    # -I_slackr + G12(V1r - V2r) - B12(V1i - V2i) = 0
    m.c1_kcl_real = pyo.Constraint(expr= 
        m.Islack_r + m.G12*(m.V1r - m.V2r) - m.B12*(m.V1i - m.V2i) == 0
    )

    # KCL at Node 1 (Imaginary)
    # -I_slacki + G12(V1i - V2i) + B12(V1r - V2r) = 0
    m.c1_kcl_imag = pyo.Constraint(expr= 
        m.Islack_i + m.G12*(m.V1i - m.V2i) + m.B12*(m.V1r - m.V2r) == 0
    )

    # --- Node 2 Constraints ---
    
    # Real Part
    # -(G12(V1r - V2r) - B12(V1i - V2i)) + (1/nt)*Gx*((1/nt)V2r - V3r) - Bx((1/nt)V2i - V3i) = 0
    # Note: I am assuming the transcription of the image term -Bx(...) distributes the (1/nt) as shown in image
    m.c2_real = pyo.Constraint(expr= 
        (m.G12*(m.V2r - m.V1r) - m.B12*(m.V2i - m.V1i)) + m.Ipx_r == 0  
    )

    # Imaginary Part
    m.c2_imag = pyo.Constraint(expr=
        -(m.G12*(m.V1i - m.V2i) + m.B12*(m.V1r - m.V2r)) + m.Ipx_i == 0
    )

    # Transformer primary current definitions
    m.c2_Ipx_real = pyo.Constraint(expr=
        m.Ipx_r + m.nt * m.Isx_r == 0 
    )
    m.c2_Ipx_imag = pyo.Constraint(expr=
        m.Ipx_i + m.nt * m.Isx_i == 0 
    )

    # Node 3aux Constraints
    m.c2_Vaux_real = pyo.Constraint(expr=
        m.nt*m.Vaux_r - m.V2r  == 0
    )
    m.c2_Vaux_imag = pyo.Constraint(expr=
        m.nt*m.Vaux_i - m.V2i  == 0
    )
    m.c2aux_real = pyo.Constraint(expr=
        m.Isx_r + (m.Gx * (m.Vaux_r - m.V3r) - m.Bx * (m.Vaux_i - m.V3i)) == 0
    )
    m.c2aux_imag = pyo.Constraint(expr=
        m.Isx_i + (m.Gx * (m.Vaux_i - m.V3i) + m.Bx * (m.Vaux_r - m.V3r)) == 0
    )

    # --- Node 3 Constraints ---
    
    # Real Part
    # -(Gx((1/nt)V2r - V3r) - Bx((1/nt)V2i - V3i)) + G34(V3r - V4r) - B34(V3i - V4i) = 0
    m.c3_real = pyo.Constraint(expr=
        m.Gx * (m.V3r - m.Vaux_r) - m.Bx * (m.V3i - m.Vaux_i) +
        m.G34 * (m.V3r - m.V4r) - m.B34 * (m.V3i - m.V4i) == 0
    )

    # Imaginary Part
    m.c3_imag = pyo.Constraint(expr=
        m.Gx*(m.V3i - m.Vaux_i) - m.Bx*(m.V3r - m.Vaux_r) +
        m.G34*(m.V3i - m.V4i) + m.B34*(m.V3r - m.V4r) == 0
    )

    # --- Node 4 Constraints (With Non-Linear Load) ---
    
    # Real Part
    # -(G34(V3r - V4r) - B34(V3i - V4i)) + (PL*V4r + QL*V4i) / (V4r^2 + V4i^2) = 0
    m.c4_real = pyo.Constraint(expr=
        (m.G34*(m.V4r - m.V3r) - m.B34*(m.V4i - m.V3i)) + 
        (m.PL*m.V4r - m.QL*m.V4i) == 0
        #(m.PL*m.V4r + m.QL*m.V4i) / (m.V4r**2 + m.V4i**2) == 0
    )

    # Imaginary Part
    # -(G34(V3i - V4i) + B34(V3r - V4r)) + (PL*V4i - QL*V4r) / (V4r^2 + V4i^2) = 0
    m.c4_imag = pyo.Constraint(expr=
        (m.G34*(m.V4i - m.V3i) + m.B34*(m.V4r - m.V3r)) + 
        (m.PL*m.V4i + m.QL*m.V4r) == 0
    )

    # =========================================================================
    # 4. OBJECTIVE
    # =========================================================================
    # The image states "min: 1". This implies a feasibility problem.
    m.obj = pyo.Objective(expr=m.V4r + m.V4i, sense=pyo.minimize)

    return m

# =========================================================================
# 5. SOLVE
# =========================================================================

if __name__ == "__main__":
    model = create_power_system_model()
    
    # Use IPOPT solver
    solver = pyo.SolverFactory('ipopt')
    
    # Solve
    results = solver.solve(model, tee=True)
    
    # Print Results
    if (results.solver.status == pyo.SolverStatus.ok) and \
       (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("\nSolution Found!")
        print(f"V1: {pyo.value(model.V1r):.4f} + j{pyo.value(model.V1i):.4f}")
        print(f"V2: {pyo.value(model.V2r):.4f} + j{pyo.value(model.V2i):.4f}")
        print(f"V3: {pyo.value(model.V3r):.4f} + j{pyo.value(model.V3i):.4f}")
        print(f"V4: {pyo.value(model.V4r):.4f} + j{pyo.value(model.V4i):.4f}")
        print(f"I_slack: {pyo.value(model.Islack_r):.4f} + j{pyo.value(model.Islack_i):.4f}")
        print(f"QL: {pyo.value(model.QL):.4f}")
    else:
        print("Solver failed to find an optimal solution.")
