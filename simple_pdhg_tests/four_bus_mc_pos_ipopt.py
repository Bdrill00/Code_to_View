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
    m.QL = pyo.Param(initialize=0.2)

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
    
    #m.V4r = pyo.Var(initialize=1.0)
    #m.V4i = pyo.Var(initialize=0.0)
    
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

    # Variables that are in the bilinear terms
    Irl_U = 1
    Irl_L = -1
    m.Irl = pyo.Var(initialize=0.0, bounds=(Irl_L, Irl_U))

    Iil_U = 1
    Iil_L = -1
    m.Iil = pyo.Var(initialize=0.0, bounds=(Iil_L, Iil_U))

    Vmag2_U = 1.21  # (1.1 pu)^2
    Vmag2_L = 0.81  # (0.9 pu)^2
    m.Vmag2 = pyo.Var(initialize=1.0, bounds=(Vmag2_L, Vmag2_U))  
    
    V4r_L = 0.25
    V4r_U = 1.25
    m.V4r = pyo.Var(initialize=1.0, bounds=(V4r_L, V4r_U))


    V4i_L = -0.75
    V4i_U = 0.75
    m.V4i = pyo.Var(initialize=0.0, bounds=(V4i_L, V4i_U))

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
        (m.G34*(m.V4r - m.V3r) - m.B34*(m.V4i - m.V3i)) + m.Irl == 0
        #(m.PL*m.V4r + m.QL*m.V4i) / (m.V4r**2 + m.V4i**2) == 0
    )

    # Imaginary Part
    # -(G34(V3i - V4i) + B34(V3r - V4r)) + (PL*V4i - QL*V4r) / (V4r^2 + V4i^2) = 0
    m.c4_imag = pyo.Constraint(expr=
        (m.G34*(m.V4i - m.V3i) + m.B34*(m.V4r - m.V3r)) + m.Iil == 0
    )

    # Vmag4
    # Perform McCormick relaxation on V4r^2 + V4i^2 

    m.V4r2 = pyo.Var(initialize=1.0)
    m.V4i2 = pyo.Var(initialize=0.0)

    m.c4_vmag = pyo.Constraint(expr=
        m.V4r2 + m.V4i2 - m.Vmag2 == 0 # 0.9 pu squared
    )

    # McCormick envelopes for V4r^2
    #
    m.cmc_v4r2_1 = pyo.Constraint(expr=
        -m.V4r2 + (2 * V4r_L * m.V4r - V4r_L * V4r_L) <= 0
    )
    m.cmc_v4r2_2 = pyo.Constraint(expr=
        -m.V4r2 + (2 * V4r_U * m.V4r - V4r_U * V4r_U) <= 0
    )
    # McCormick envelopes for V4i^2
    m.cmc_v4r2_3 = pyo.Constraint(expr=
        m.V4r2 - (V4r_L * m.V4r + V4r_U * m.V4r - V4r_L * V4r_U) <= 0
    )

    # Bounds for V4i
    m.cmc_v4i2_1 = pyo.Constraint(expr=
        -m.V4i2 + (2 * V4i_L * m.V4i - V4i_L * V4i_L) <= 0
    )
    m.cmc_v4i2_2 = pyo.Constraint(expr=
        -m.V4i2 + (2 * V4i_U * m.V4i - V4i_U * V4i_U) <= 0
    )
    m.cmc_v4i2_3 = pyo.Constraint(expr=
        m.V4i2 - (V4i_L * m.V4i + V4i_U * m.V4i - V4i_L * V4i_U) <= 0
    )
    # Load current definitions
    #m.c4_load_real = pyo.Constraint(expr=
    #    m.Irl*m.Vmag2 - (m.PL*m.V4r + m.QL*m.V4i) == 0
    #)
    #m.c4_load_imag = pyo.Constraint(expr=
    #    m.Iil*m.Vmag2 - (m.PL*m.V4i - m.QL*m.V4r) == 0
    #)

    # --- 2. Create Auxiliary Variables ---
    # These represent the result of the multiplication: I * Vmag2
    m.W_real = pyo.Var(initialize=1.0) # Represents Irl * Vmag2
    m.W_imag = pyo.Var(initialize=0.0) # Represents Iil * Vmag2

    # --- 3. The Linearized Load Constraints ---
    # Replaced 'm.Irl*m.Vmag2' with 'm.W_real'
    m.c4_load_real = pyo.Constraint(expr=
        m.W_real - (m.PL*m.V4r + m.QL*m.V4i) == 0
    )

    # Replaced 'm.Iil*m.Vmag2' with 'm.W_imag'
    m.c4_load_imag = pyo.Constraint(expr=
        m.W_imag - (m.PL*m.V4i - m.QL*m.V4r) == 0
    )

    # --- 4. McCormick Envelopes for W_real (Irl * Vmag2) ---
    # z = x * y  =>  z = W_real, x = Irl, y = Vmag2

    # Underestimator 1
    m.cmc_load_real_1 = pyo.Constraint(expr=
        -m.W_real + Irl_L * m.Vmag2 + Vmag2_L * m.Irl - Irl_L * Vmag2_L <= 0
    )
    # Underestimator 2
    m.cmc_load_real_2 = pyo.Constraint(expr=
        -m.W_real + Irl_U * m.Vmag2 + Vmag2_U * m.Irl - Irl_U * Vmag2_U <= 0
    )
    # Overestimator 1
    m.cmc_load_real_3 = pyo.Constraint(expr=
        m.W_real - (Irl_U * m.Vmag2 + Vmag2_L * m.Irl - Irl_U * Vmag2_L) <= 0
    )
    # Overestimator 2
    m.cmc_load_real_4 = pyo.Constraint(expr=
        m.W_real - (Irl_L * m.Vmag2 + Vmag2_U * m.Irl - Irl_L * Vmag2_U) <= 0
    )

     
    # --- 5. McCormick Envelopes for W_imag (Iil * Vmag2) ---
    # z = x * y  =>  z = W_imag, x = Iil, y = Vmag2

    # Underestimator 1
    m.cmc_load_imag_1 = pyo.Constraint(expr=
        -m.W_imag + Iil_L * m.Vmag2 + Vmag2_L * m.Iil - Iil_L * Vmag2_L <= 0
    )
    # Underestimator 2
    m.cmc_load_imag_2 = pyo.Constraint(expr=
        -m.W_imag + Iil_U * m.Vmag2 + Vmag2_U * m.Iil - Iil_U * Vmag2_U <= 0
    )
    # Overestimator 1
    m.cmc_load_imag_3 = pyo.Constraint(expr=
        m.W_imag - (Iil_U * m.Vmag2 + Vmag2_L * m.Iil - Iil_U * Vmag2_L ) <= 0
    )
    # Overestimator 2
    m.cmc_load_imag_4 = pyo.Constraint(expr=
        m.W_imag - (Iil_L * m.Vmag2 + Vmag2_U * m.Iil - Iil_L * Vmag2_U) <= 0
    )

    # =========================================================================
    # 4. OBJECTIVE
    # =========================================================================
    # The image states "min: 1". This implies a feasibility problem.
    m.obj = pyo.Objective(expr=-m.V4r, sense=pyo.minimize)

    return m

# =========================================================================
# 5. SOLVE
# =========================================================================

def return_model():
    return create_power_system_model()

if __name__ == "__main__":
    model = create_power_system_model()

    # Use IPOPT solver
    solver = pyo.SolverFactory('gurobi')
    
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
    else:
        print("Solver failed to find an optimal solution.")
