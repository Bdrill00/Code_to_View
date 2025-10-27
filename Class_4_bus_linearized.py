import math
import numpy as np
import jax
import jax.numpy as jnp
""" 
It is helpful to note the following
1. Voltage variables go from 1-n and are of the order V1r, v1i, v2r, v2i,....
2. Current variables are the same
3. McCormick variables are x0, y0, z0, x1, y1, z1, v, w|x2, y2... which is real variables of McCormick, imaginary, then the variables with the real and imaginary voltages squared
"""

class Line:
    all_lines = []
    def __init__(self, name, from_node, to_node, length, phases, Imp_mat):
        self.id = name
        self.from_node = from_node
        self.to_node = to_node
        self.line_length = length
        self.phases = phases
        self.c_type = 'Line'
        self.imp_mat = Imp_mat
        Line.all_lines.append(self)
        #What I would like to do here is take the real and imaginary voltages from the to and from nodes and create a matrix
        #showing the directional current flow
    def admit_mat(self, in_out):
        zLine = self.imp_mat*self.line_length
        yLine = jnp.linalg.inv(zLine)
        G_line = yLine.real
        B_line = yLine.imag
        if in_out == 'out':
            #                    n1r      n2r      n1i      n2i
            node_1 = jnp.block([[G_line,-B_line],
                                [B_line, G_line]])
            node_2 = jnp.block([[-G_line,B_line],
                                [-B_line,-G_line]])
        elif in_out == 'in':
            #                    n1r      n2r      n1i      n2i
            node_1 = jnp.block([[-G_line,B_line,],
                                [-B_line,-G_line]])
            
            node_2 = jnp.block([[G_line,-B_line],
                                [B_line,G_line]])
        else:
            print('error')
        return node_1, node_2    
    def __repr__(self):
        base = {"id": self.id, "from_node": self.from_node,"to_node": self.to_node,"length": self.line_length,"phases": self.phases}
        return str(base)

class Transformer:
    all_trans = []
    def __init__(self, name, from_node, to_node, t_type, n_of_t, T_rating, Volt_Rate, Z_PU):
        self.id = name
        self.from_node = from_node
        self.to_node = to_node
        self.transformer_type = t_type
        self.nt = n_of_t
        self.t_rate = T_rating
        self.c_type = 'Transformer'
        self.vRate = Volt_Rate
        self.zpu = Z_PU
        Transformer.all_trans.append(self)
    def admit_mat(self, in_out):
        Ztbase = (self.vRate)**2/self.t_rate
        zt = Ztbase*self.zpu
        Ad_mat = zt*jnp.eye(3)
        Ytr = jnp.linalg.inv(Ad_mat)
        Gtr = Ytr.real
        Btr = Ytr.imag
        one_nt_sqr = 1/(self.nt)**2
        one_nt = 1/self.nt
        if in_out == 'out':
            T_flow_node_1 = jnp.block([[one_nt_sqr*Gtr,-one_nt_sqr*Btr],
                                       [one_nt_sqr*Btr,one_nt_sqr*Gtr]])
            T_flow_node_2 = jnp.block([[-one_nt*Gtr,one_nt*Btr],
                                       [-one_nt*Btr,-one_nt*Gtr]])
        elif in_out == 'in':
            T_flow_node_1 = jnp.block([[-one_nt*Gtr, one_nt*Btr],
                                       [-one_nt*Btr,-one_nt*Gtr]])
            T_flow_node_2 = jnp.block([[Gtr,-Btr],
                                       [Btr,Gtr]])
        else:
            print('error1')
        return T_flow_node_1, T_flow_node_2
    
    def __repr__(self):
        base = {"id": self.id, "from_node": self.from_node,"to_node": self.to_node,"Transformer Type": self.transformer_type,"Turns Ratio": self.nt}
        return str(base)
        

class Load:
    all_loads = []
    def __init__(self, name, from_node, realP_vec, pf_Vec,phases):
        self.id = name
        self.from_node = from_node
        self.realP = jnp.array(realP_vec, dtype=float)
        self.pf = jnp.array(pf_Vec, dtype=float)
        angle = jnp.arccos(self.pf)
        self.imagQ = realP_vec*jnp.tan(angle)
        self.c_type = 'Load'
        self.load_num = len(Load.all_loads) + 1
        self.phases = phases
        Load.all_loads.append(self)
    def current_matrix(self, in_out):#, Vr_init, Vi_init):
        num_p = len(self.phases)
        b = jnp.zeros((6,1))
        # dvrdt = jnp.zeros((6,1))
        # dvidt = dvrdt.copy()
        # for i in range(len(self.phases)):
        #     eq1 = (self.realP[i]*Vr_init[i]+self.imagQ[i]*Vi_init[i])/(Vr_init**2+Vi_init**2)
        #     eq2 = (self.realP[i]*Vi_init[i]-self.imagQ[i]*Vr_init[i])/(Vr_init**2+Vi_init**2)
        #     eq3 = (self.realP[i]*(Vr_init[i]**2+Vi_init**2)-2*(self.realP[i]*Vr_init[i]+self.imagQ[i]*Vi_init[i])*Vr_init[i])/(Vr_init**2+Vi_init**2)**2
        #     eq4 = (self.realQ[i]*(Vr_init[i]**2+Vi_init**2)-2*(self.realP[i]*Vr_init[i]+self.imagQ[i]*Vi_init[i])*Vi_init[i])/(Vr_init**2+Vi_init**2)**2
        #     eq5 = (-self.realQ[i]*(Vr_init[i]**2+Vi_init**2)-2*(self.realP[i]*Vr_init[i]+self.imagQ[i]*Vi_init[i])*Vr_init[i])/(Vr_init**2+Vi_init**2)**2
        #     eq6 = (self.realP[i]*(Vr_init[i]**2+Vi_init**2)-2*(self.realP[i]*Vr_init[i]+self.imagQ[i]*Vi_init[i])*Vi_init[i])/(Vr_init**2+Vi_init**2)**2
        #     constr = eq1-eq3*Vr_init[i]-eq4*Vi_init[i]
        #     consti = eq2-eq5*Vr_init[i]-eq6*Vi_init[i]
        #     b = b.at[i].set(constr)
        #     b = b.at[i+num_p].set(consti)
        #     dvrdt = dvrdt.at[i].set(eq3)
        #     dvrdt = dvrdt[i+num_p].set(eq5)
        #     dvidt = dvidt[i].set(eq4)
        #     dvidt = dvidt[i+num_p].set(eq6)
        
            #this will output vector b and var which we need to convert into a matrix
        if in_out == 'in':
            Load_mat = jnp.block([[-jnp.eye(num_p), jnp.zeros((num_p,num_p))],
                                [jnp.zeros((num_p,num_p)), -jnp.eye(num_p)]])
            # Load_mat2 = jnp.block([[jnp.diag(dvrdt[0:num_p,1]), jnp.diag(dvidt[0:num_p,1])],
            #                        [jnp.diag(dvrdt[num_p:2*num_p,1]), jnp.eye(num_p)@dvidt[num_p:num_p,1]]])
            b_out = b
        elif in_out == 'out':
            Load_mat = jnp.block([[jnp.eye(num_p), jnp.zeros((num_p,num_p))],
                                [jnp.zeros((num_p,num_p)), jnp.eye(num_p)]])
            # Load_mat2 = jnp.block([[-jnp.eye(num_p)@dvrdt[0:num_p,1], -jnp.eye(num_p)@dvidt[0:num_p,1]],
            #                        [-jnp.eye(num_p)@dvrdt[num_p:2*num_p,1], -jnp.eye(num_p)@dvidt[num_p:num_p,1]]])
            # b_out = -b
        else:
            print('error')
        return Load_mat
        # return Load_mat2, b_out
    def McCormick(self):
        return
    def __repr__(self):
        base = {"id": self.id, "from_node": self.from_node}
        return str(base)
        
class Generator:
    all_gen = []
    def __init__(self, name, to_node, voltage, phases):
        self.id = name
        self.to_node = to_node
        self.voltage = voltage
        self.phases = phases
        self.c_type = 'Generator'
        self.gen_num = len(Generator.all_gen) + 1
        Generator.all_gen.append(self)
    def current_matrix(self, in_out):
        #if we consider current leaving and entering
        if in_out == 'in':
            Gen_mat = jnp.block([[-jnp.eye(3), jnp.zeros((3,3))],
                                [jnp.zeros((3,3)), -jnp.eye(3)]])
        elif in_out == 'out':
            Gen_mat = jnp.block([[jnp.eye(3), jnp.zeros((3,3))],
                                [jnp.zeros((3,3)), jnp.eye(3)]])
        else:
            print('error')
        return Gen_mat
    def __repr__(self):
        base = {"id": self.id, "to_node": self.to_node}
        return str(base)
        
def connected_to(node):
    connected = []
    for line in Line.all_lines:
        if line.from_node == node or line.to_node == node:
            connected.append(line)
    for t in Transformer.all_trans:
        if t.from_node == node or t.to_node == node:
            connected.append(t)
    for load in Load.all_loads:
        if load.from_node == node:
            connected.append(load)
    for gen in Generator.all_gen:
        if gen.to_node == node:
            connected.append(gen)
    return connected

def powerflow(nodes,variables,phases,positions):
    """what do I need to know in order to construct this matrix...
    The node corresponds directly to columns being used
    The reals are every three: V1r is 0:2 columns, V2r is 6:8,...ending with 6*#ofnodes = 24.
    at 24 we have the generation current variables, at 30 we have the load
    lets think this out, if i am at node 2, we can easily count # of equations to figure out the row
    but node 2 will correspond to starting (2-1)*6 = 6, generators will have to go by the number assigned to them
    but that will be 24+(i-1)*6, loads will be the same as that, it will start at the column which is equal
    to the number of total nodes and total generator variables (real, imag, per phase).  Note, when refering to nodal voltage,
    V1 or generator current Ig1, it is important to note that each variable, in this system contains 3 phases and a real and
    imaginary part per each phase.
    
    Per each node, I have already created a function which groups my classes together at each node, now the process I want to follow is as follows:
    1. I want to read the type of each instance is.  Use this later to figure out where to stamp the data
    2. I want to read if it has a 'from_node' and if that 'from_node' is equal to the node we are on
    3. Same for to_node.  I need to do it this way because generator and load class are missing from_node and to_node respectively
    I think for line of the mccormick constraint I will do each phase before moving to the enxt
    """
    A_mat = jnp.zeros((0,len(variables)))
    b_vec = jnp.zeros((0,1))
    for i in range(1, nodes+1):
        connected_to_1 = connected_to(i)
        # print(f"Components connected to node {i}:")
        #For each instance connected to a specified node:
        eq_mat = jnp.zeros((2*len(phases), len(variables)))
        b_out = jnp.zeros((2*len(phases),1))
        for c in connected_to_1:
            node_mat = jnp.zeros((2*len(phases), len(variables)))
            instance = c
            # print(instance)
            # print(instance.id)
            #if that node has a "from_key"...trying to isolate things to input into my matrix
            #This is important for figuring out which matrix to choose from each class' respective method
            if hasattr(instance, "from_node"):
                if instance.from_node == i:
                    #this isolates down to transformer, line, and load
                    #Now this is where I extract the admittance or current matrix from the respective instance to put into my A matrix
                    if instance.c_type == 'Line':
                        line_mat1, line_mat2 = instance.admit_mat(in_out = 'out')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(line_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(line_mat2)
                        b = jnp.zeros((2*len(phases),1))
                    elif instance.c_type == 'Transformer':
                        t_mat1, t_mat2 = instance.admit_mat('out')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(t_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(t_mat2)
                        b = jnp.zeros((2*len(phases),1))
                    elif instance.c_type == 'Load':
                        l_mat = instance.current_matrix('out')
                        index_val = positions['loads'][0] + 6*(instance.load_num - 1)
                        node_mat = node_mat.at[:,index_val:index_val+6].set(l_mat)
                        b = jnp.zeros((2*len(phases),1))
                #for current going into the node #Have not verified this commented out elif for linearized load w/ taylor
                    # elif instance.c_type == 'Load':     
                    #     VMag = jnp.array([1918, 2061, 1981])
                    #     VAngle = jnp.array([-9.1, -128.3, 110.9])
                    #     vr = jnp.zeros((VMag.shape[0], 1))
                    #     vi = jnp.zeros((VMag.shape[0], 1))
                    #     for i in range(VMag.shape[0]):
                    #         vr[i] = VMag[i]*np.cos(math.radians(VAngle[i]))
                    #         vi[i] = VMag[i]*np.sin(math.radians(VAngle[i]))
                    #     l_mat,b_mat = instance.current_matrix('out', vr,vi,phases)
                    #     index_val1 = 6*(instance.from_node-1)
                    #     node_mat = node_mat.at[:,index_val1+6].set(l_mat)
                    #     b = b_mat 
            if hasattr(instance, "to_node"):
                if instance.to_node == i:
                    if instance.c_type == 'Line':
                        line_mat1, line_mat2 = instance.admit_mat('in')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(line_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(line_mat2)
                        b = jnp.zeros((2*len(phases),1))
                    elif instance.c_type == 'Transformer':
                        t_mat1, t_mat2 = instance.admit_mat('in')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(t_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(t_mat2)
                        b = jnp.zeros((2*len(phases),1))
                    elif instance.c_type == 'Generator':
                        gen_mat = instance.current_matrix('in')
                        index_val = positions['generators'][0]+6*(instance.gen_num - 1)
                        node_mat = node_mat.at[:,index_val:index_val+6].set(gen_mat)
                        b = jnp.zeros((2*len(phases),1))
                        # print(line_mat1, line_mat2)
            eq_mat = eq_mat + node_mat
            b_out = b_out+b
        A_mat = jnp.vstack([A_mat,eq_mat])
        b_vec = jnp.vstack([b_vec,b_out])
    return A_mat,b_vec
