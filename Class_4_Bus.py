import math
import numpy as np
import jax
import jax.numpy as jnp
class Line:
    def __init__(self, name, to_node, from_node, length, phases):
        self.id = name
        self.to_node = to_node
        self.from_node = from_node
        self.line_length = length
        self.phases = len(phases)
        #What I would like to do here is take the real and imaginary voltages from the to and from nodes and create a matrix
        #showing the directional current flow
    def admit_mat(self, Imp_mat):
        zLine = Imp_mat*self.line_length
        yLine = jnp.linalg.inv(zLine)
        G_line = yLine.real
        B_line = yLine.imag
        R_q_biflow = jnp.block([[G_line, -G_line, -B_line, B_line],
                               [B_line, -B_line, G_line, -G_line],
                               [-G_line, G_line, B_line, -B_line],
                               [-B_line, B_line, -G_line, G_line]])
        return R_q_biflow    
    def __repr__(self):
        base = {"id": self.id, "from_node": self.from_node,"to_node": self.to_node,"length": self.line_length,"phases": self.phases}
        return str(base)




class Transformer:
    def __init__(self, name, to_node, from_node, t_type, n_of_t):
        self.id = name
        self.to_node = to_node
        self.from_node = from_node
        self.transformer_type = t_type
        self.nt = n_of_t
    def admit_mat(self, T_rating, Volt_Rate, Z_PU):
        Ztbase = (Volt_Rate)**2/T_rating
        zt = Ztbase*Z_PU
        Ad_mat = zt*jnp.eye(3)
        Ytr = jnp.linalg.inv(Ad_mat)
        Gtr = Ytr.real
        Btr = Ytr.imag
        T_flow_mat = jnp.block([[self.nt*Gtr, -Gtr, -self.nt*Btr, Btr],
                               [self.nt*Btr, Btr, self.nt*Gtr, -Gtr],
                               [-self.nt*Gtr, Gtr, self.nt*Btr, -Gtr],
                               [-self.nt*Btr, Btr, -self.nt*Gtr, Gtr]])
        return T_flow_mat    
        
        

class Load:
    def __init__(self, name, from_node, realP_vec, imagQ_vec, pf_Vec):
        self.id = name
        self.from_node = from_node
        self.realP = jnp.array(realP_vec, dtype=float)
        self.pf = jnp.array(pf_Vec, dtype=float)
        angle = jnp.arccos(self.pf)
        self.imagQ = realP_vec*jnp.tan(angle)

class Generator:
    def __init__(self, name, to_node, voltage, phases):
        self.id = name
        self.to_node = to_node
        self.voltage = voltage
        self.phases = phases
        



