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
    def __init__(self, name, from_node, realP_vec, pf_Vec):
        self.id = name
        self.from_node = from_node
        self.realP = jnp.array(realP_vec, dtype=float)
        self.pf = jnp.array(pf_Vec, dtype=float)
        angle = jnp.arccos(self.pf)
        self.imagQ = realP_vec*jnp.tan(angle)
        self.c_type = 'Load'
        self.load_num = len(Load.all_loads) + 1
        Load.all_loads.append(self)
    def current_matrix(self, in_out):
        if in_out == 'in':
            Load_mat = jnp.block([[-jnp.eye(3), jnp.zeros((3,3))],
                                [jnp.zeros((3,3)), -jnp.eye(3)]])
        elif in_out == 'out':
            Load_mat = jnp.block([[jnp.eye(3), jnp.zeros((3,3))],
                                [jnp.zeros((3,3)), jnp.eye(3)]])
        else:
            print('error')
        return Load_mat
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

# def McCor_relax(XL, XU, YL, YU, phases):
#     """What will this return?  This will return the 3-phase McCormick relaxation of 1 bilinear term 
#     XU*YU >= YU*x + XU*y - z,
#     -XU*YL >= z - YL*x - XU*y,
#     -XL*YU >= z - YU*x - XL*y ,
#     XL*YL >= YL*x + XL*y - z
#     """
#     n = len(phases)
#     b = jnp.zeros((4*n, 1))
#     x = jnp.zeros((4*n, n))
#     y = x.copy()
#     z = x.copy()
#     const = jnp.zeros((n*4,4))
#     for indx, p in enumerate(phases):
#         const = const.at[indx,:].set(jnp.array([float(XU[indx][0]*YU[indx][0]), float(YU[indx][0]), float(XU[indx][0]), -1]))
#         const = const.at[n+indx,:].set(jnp.array([float(-XU[indx][0]*YL[indx][0]), float(-YL[indx][0]), float(-XU[indx][0]), 1]))
#         const = const.at[2*n+indx,:].set(jnp.array([float(-XL[indx][0]*YU[indx][0]), float(-YU[indx][0]), float(-XL[indx][0]), 1]))
#         const = const.at[3*n+indx,:].set(jnp.array([float(XL[indx][0]*YL[indx][0]), float(YL[indx][0]), float(XL[indx][0]), -1]))
#     # const = jnp.array([[XU*YU, YU, XU, -1],
#     #                     [-XU*YL, -YL, -XU, 1],
#     #                     [-XL*YU, -YU, -XL, 1],
#     #                     [XL*YL, YL, XL, -1]])
#     listVar = [b, x, y, z]
#     print(const.shape)
#     for indx, a in enumerate(listVar):
#         if a.all() == b.all():
#             listVar[indx] = a.at[0:n, :].set(const[0:n,indx].reshape(-1,1))
#             listVar[indx] = a.at[n:2*n,:].set(const[n:2*n,indx].reshape(-1,1))
#             listVar[indx] = a.at[2*n:3*n,:].set(const[2*n:3*n,indx].reshape(-1,1))
#             listVar[indx] = a.at[3*n:4*n, :].set(const[3*n:4*n,indx].reshape(-1,1))
#         else:
#             listVar[indx] = a.at[0:n, :].set(jnp.ones((n,a.shape[1]))@const[0:n,indx].reshape(-1,1))
#             listVar[indx] = a.at[n:2*n,:].set(jnp.ones((n,a.shape[1]))@const[n:2*n,indx].reshape(-1,1))
#             listVar[indx] = a.at[2*n:3*n,:].set(jnp.ones((n,a.shape[1]))@const[2*n:3*n,indx].reshape(-1,1))
#             listVar[indx] = a.at[3*n:4*n, :].set(jnp.ones((n,a.shape[1]))@const[3*n:4*n,indx].reshape(-1,1))    
        
#     return listVar[0], listVar[1], listVar[2], listVar[3]
def McCor_relax(XL, XU, YL, YU, phases):
    n = len(phases)

    XL = XL.flatten()
    XU = XU.flatten()
    YL = YL.flatten()
    YU = YU.flatten()
    b = jnp.zeros((4*n,1))
    x = jnp.zeros((4*n,n))
    y = jnp.zeros((4*n,n))
    z = jnp.zeros((4*n,n))
    const = jnp.zeros((4*n,4))
    for i in range(n):
        const = const.at[i,:].set(jnp.array([XU[i]*YU[i], YU[i], XU[i], -1]))
        const = const.at[n+i,:].set(jnp.array([-XU[i]*YL[i], -YL[i], XU[i], 1]))
        const = const.at[2*n+i,:].set(jnp.array([-XL[i]*YU[i], -YU[i], XL[i], 1]))
        const = const.at[3*n+i,:].set(jnp.array([XL[i]*YL[i], YL[i], XL[i], -1]))
    for i in range(4):
        b = b.at[i*n:(i+1)*n,0].set(const[i*n:(i+1)*n,0])
        x = x.at[i*n:(i+1)*n,:].set(jnp.eye(n) * const[i*n:(i+1)*n,1:2])
        y = y.at[i*n:(i+1)*n,:].set(jnp.eye(n) * const[i*n:(i+1)*n,2:3])
        z = z.at[i*n:(i+1)*n,:].set(jnp.eye(n) * const[i*n:(i+1)*n,3:4])
    
    return b, x, y, z

def Quad_relax(XL, XU, phases):
    """
    XU**2 >= 2*XU*x - y,
    XL**2 >= 2*x*XL - y,
    -XL*XU>= y - (XU+XL)*x 
    """
    n = len(phases)
    XL = XL.flatten()
    XU = XU.flatten()
    b = jnp.zeros((3*n, 1))
    x = jnp.zeros((3*n, n))
    y = x.copy()
    const = jnp.zeros((3*n,3))
    for i in range(n):
        const = const.at[i,:].set(jnp.array([XU[i]**2, 2*XU[i], -1]))
        const = const.at[n+i,:].set(jnp.array([XL[i]**2, 2*XL[i], -1]))
        const = const.at[2*n+i,:].set(jnp.array([-XL[i]*XU[i], -(XU[i]+XL[i]), 1]))
    for i in range(3):
        b = b.at[i*n:(i+1)*n,0].set(const[i*n:(i+1)*n,0])
        x = x.at[i*n:(i+1)*n,:].set(jnp.eye(n) * const[i*n:(i+1)*n,1:2])
        y = y.at[i*n:(i+1)*n,:].set(jnp.eye(n) * const[i*n:(i+1)*n,2:3])
    return b, x, y

        
        
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

    for i in range(1, nodes+1):
        connected_to_1 = connected_to(i)
        # print(f"Components connected to node {i}:")
        #For each instance connected to a specified node:
        eq_mat = jnp.zeros((2*len(phases), len(variables)))
        for c in connected_to_1:
            node_mat = jnp.zeros((2*len(phases), len(variables)))
            instance = c
            print(instance)
            print(instance.id)
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
                    elif instance.c_type == 'Transformer':
                        t_mat1, t_mat2 = instance.admit_mat('out')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(t_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(t_mat2)
                    elif instance.c_type == 'Load':
                        l_mat = instance.current_matrix('out')
                        index_val = positions['loads'][0] + 6*(instance.load_num - 1)
                        node_mat = node_mat.at[:,index_val:index_val+6].set(l_mat)
                #for current going into the node
            if hasattr(instance, "to_node"):
                if instance.to_node == i:
                    if instance.c_type == 'Line':
                        line_mat1, line_mat2 = instance.admit_mat('in')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(line_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(line_mat2)
                    elif instance.c_type == 'Transformer':
                        t_mat1, t_mat2 = instance.admit_mat('in')
                        index_val1 = 6*(instance.from_node - 1)
                        index_val2 = 6*(instance.to_node - 1)
                        node_mat = node_mat.at[:,index_val1:index_val1+6].set(t_mat1)
                        node_mat = node_mat.at[:,index_val2:index_val2+6].set(t_mat2)
                    elif instance.c_type == 'Generator':
                        gen_mat = instance.current_matrix('in')
                        index_val = positions['generators'][0]+6*(instance.gen_num - 1)
                        node_mat = node_mat.at[:,index_val:index_val+6].set(gen_mat)
                        # print(line_mat1, line_mat2)
            eq_mat = eq_mat + node_mat
        A_mat = jnp.vstack([A_mat,eq_mat])
    return A_mat
""" 
P*V4r + QV4i = z(V4r^2+V4i^2) = z(w+v) = x+y 
x = zv
y = zw
------
P*V4i + QV4r = z1(V4r^2+V4i^2) = z1(w+v) = x1+y1
x1 = z1v
y1 = z1w
------
v = V4r^2
w = V4i^2
"""
def MC_Stack(temp_mat, big_b,position_matrix,row1,xinit,yinit,zinit,x,y,z,b,phs):
    x_block = x.reshape(phs,-1)
    y_block = y.reshape(phs,-1)
    z_block = z.reshape(phs,-1)
    b = b.reshape(phs,-1)
    for i in range(4):
        temp_mat = temp_mat.at[row1:row1+phs, position_matrix[0]+xinit*phs:(position_matrix[0]+xinit*phs)+phs].set(jnp.eye(phs)@x_block[:,i])
        temp_mat = temp_mat.at[row1:row1+phs, position_matrix[0]+yinit*phs:(position_matrix[0]+yinit*phs)+phs].set(jnp.eye(phs)@y_block[:,i])
        temp_mat = temp_mat.at[row1:row1+phs, position_matrix[0]+zinit*phs:(position_matrix[0]+zinit*phs)+phs].set(jnp.eye(phs)@z_block[:,i])
        big_b = big_b.at[row1:row1+phs,:].set(b[phs*i:phs*(i+1),:])
        row1 +=phs
    return temp_mat, big_b
def MC_Stack(temp_mat, big_b, position_matrix, row1, xinit, yinit, zinit, x, y, z, b, phs):
    col_x = position_matrix[0] + xinit * phs
    col_y = position_matrix[0] + yinit * phs
    col_z = position_matrix[0] + zinit * phs

    temp_mat = temp_mat.at[row1:row1+4*phs, col_x:col_x+phs].set(x)
    temp_mat = temp_mat.at[row1:row1+4*phs, col_y:col_y+phs].set(y)
    temp_mat = temp_mat.at[row1:row1+4*phs, col_z:col_z+phs].set(z)
    big_b = big_b.at[row1:row1+4*phs, :].set(b)
    return temp_mat, big_b
def Quad_Stack(temp_mat,big_b, position_matrix,row1,xinit,yinit,x,y,b,phs):
    temp_mat = temp_mat.at[row1:row1+3*phs,xinit:xinit+phs].set(x)
    temp_mat = temp_mat.at[row1:row1+3*phs,position_matrix[0]+yinit*phs:(position_matrix[0]+yinit*phs)].set(y)
    big_b = big_b.at[row1:row1+3*phs,:].set(b)
    return temp_mat, big_b

def McC_Load(upperBs, lowerBs, variables, phases, position_matrix,load_node):
    node_L = load_node
    V_indx = variables.index(f"v{node_L}ra")
    phs = len(phases)
    temp_mat = jnp.zeros((phs*16+phs*6,len(variables)))
    big_b = jnp.zeros((12*4+9*2,1))
    # num_loads = len(Load.all_loads)
    #key for each of the variables:xr, yr, zr, zi, yi, zi, vi, wi are numbered 0-7
    b1, zr1, v1, xr1, = McCor_relax(lowerBs[0:3,:], upperBs[0:3,:], lowerBs[6:9,:], upperBs[6:9,:],phases)
    temp_mat, big_b = MC_Stack(temp_mat, big_b, position_matrix,(0*4*phs),2,6,0,zr1,v1,xr1,b1,phs)
    # b2, zr2, w1, yr1, = McCor_relax(lowerBs[0:3,:], upperBs[0:3,:], lowerBs[9:12,:], upperBs[9:12,:],phases)
    # temp_mat, big_b = MC_Stack(temp_mat, big_b,position_matrix,1*4*phs,2,7,1,zr2, w1, yr1,b2,phs)
    # b3, zi1, v2, xi2, = McCor_relax(lowerBs[3:6,:], upperBs[3:6,:], lowerBs[6:9,:], upperBs[6:9,:],phases)
    # temp_mat, big_b = MC_Stack(temp_mat, big_b,position_matrix,2*4*phs,5,6,3,zi1,v2,xi2,b3,phs)
    # b4, zi2, w2, yi2, = McCor_relax(lowerBs[3:6,:], upperBs[3:6,:], lowerBs[9:12,:], upperBs[9:12,:],phases)
    # temp_mat, big_b = MC_Stack(temp_mat, big_b,position_matrix,3*4*phs,5,7,4,zi2,w2,yi2,b4,phs)
    # b5, Vr, v3 = Quad_relax(lowerBs[12:15,0], upperBs[12:15,0], phases)
    # temp_mat, big_b = Quad_Stack(temp_mat, big_b, position_matrix,4*4*phs,V_indx, 6, Vr, v3, b5,phs)
    # b6, Vi, w3 = Quad_relax(lowerBs[15:18,0], upperBs[15:18,0], phases)
    # temp_mat, big_b = Quad_Stack(temp_mat, big_b, position_matrix,4*4*phs+3*phs,V_indx+phs, 7, Vi, w3, b6,phs)
        #order xr, yr, zr, xi, yi, zi, v, w
        #have to figure out how to stamp Vr, Vi

    return temp_mat, big_b

def init_func(node,phases,Vupper,variables):
    start_node = variables.index(f"v{node}ra")
    phs = len(phases)
    Vsr = jnp.array([
        [Vupper * jnp.cos(jnp.radians(0))],
        [Vupper * jnp.cos(jnp.radians(-120))],
        [Vupper * jnp.cos(jnp.radians(120))]
    ])
    Vsi = jnp.array([
        [Vupper * jnp.sin(jnp.radians(0))],
        [Vupper * jnp.sin(jnp.radians(-120))],
        [Vupper * jnp.sin(jnp.radians(120))]
    ])
    init_matrix = jnp.zeros((6,len(variables)))
    init_matrix = init_matrix.at[start_node:start_node+phs,start_node:start_node+phs].set(jnp.eye(3)@jnp.diag(Vsr.flatten()))
    init_matrix = init_matrix.at[start_node+phs:start_node+2*phs,start_node+phs:start_node+2*phs].set(jnp.diag(Vsi.flatten()))
    return init_matrix



inits = 1*18
uppers = jnp.zeros((18,1))
lowers = uppers.copy()
tot_upper = jnp.zeros((inits,1))
tot_lower = tot_upper.copy()
for n in range(1, 1 +1):
    uppers = uppers.at[0:6,:].set(1200)
    lowers = lowers.at[0:6,:].set(-1200)
    uppers = uppers.at[6:12,:].set(4160**2)
    lowers = lowers.at[6:12,:].set(0)
    uppers = uppers.at[12:18,:].set(4160)
    lowers = lowers.at[12:18,:].set(-4160)
    tot_upper = tot_upper.at[18*(n-1):18*n,:].set(uppers)
    tot_lower = tot_lower.at[18*(n-1):18*n,:].set(lowers)
# b5, V4r, v3 = Quad_relax(tot_lower[12:15,0], tot_upper[12:15,0], ['a','b','c'])
# print(b5)
# print(V4r)
# print(v3.shape)
b1, zr1, v1, xr1, = McCor_relax(tot_lower[0:3,:], tot_upper[0:3,:], tot_lower[6:9,:], tot_upper[6:9,:], ['a','b','c'])
# print('b',b1)
# print('zr1', zr1)
# print('v1',v1)
# print('xr1',xr1.shape)
"""
So I know that for my load, that I have two real mccormick relaxations, to imaginary, and two that are quadratic 
and real.  Per each load I would like to stack them as such


I am a little stuck here, not sure how to go about this, my thoughts are that I should either define this function
based off of one single McCormick constraint and then iterate over it for each phase, real and imaginary part, and
each load.  However, the load index corresponds directly with the McCormick index.  I need to know how many
variables are created for one load.  Per load I have 6 single phase relaxation equations.  Two of those equations 
are QUADRATIC NOT BILINEAR so I would do those in a seperate function.  That leaves me with four equations.  Two 
of those equations are real and two are imaginary.  So the order of those equations will be the same as the
variables; the reals will be first and then the imaginary. For each of those equations, we have 4 equations and
each variable is actually 3 phases.  Thus we have (4 single phase equations)*(3 phases)*(4 McCormick inequalities
per each single phase equation) leaving with a grand total of 48 equations.  We have only 2 quadratic equality
constraints, one real one imaginary, and three phases with three equations per single phase representation: this
gives 2*3*3 = 18 -> 18+48 = 66 total equations.  So that means my index value is going to be
index_var = position[McVar]+66(load.load_number - 1).

What I am not so sure about is how to organize the iteration of stamping the mccormick iterations in the loop.
There are a few options, I will list them but then I have to pick a lane and drive.Lets go with seperating by 
P*V4r + QV4i = z(V4r^2+V4i^2) = z(w+v) = x+y 
x = zw
y = zv
------
P*V4i + QV4r = z1(V4r^2+V4i^2) = z1(w+v) = x1+y1
x1 = z1w
y1 = z1w
------
v = V4r^2
w = V4i^2
I think I should make a McCormick class which makes this more callable
"""
#we know that the voltage used for the load corresponds to the voltage at the bus its connected to
#where we input the mccormick constraints into our matrix depends on which load we are on.
        
# phases = ['a','b','c']
# num_nodes = 4
# generators = 1
# loads = 1

# variables = []  # combined variable list
# positions = {}  # dictionary to store ranges

# # Node voltages
# start = len(variables)
# for n in range(1, num_nodes+1):
#     for p in phases:
#         variables.append(f"v{n}r{p}")
#     for p in phases:
#         variables.append(f"v{n}i{p}")
# end = len(variables)
# positions['lines'] = (start, end)

# # Generator currents
# start = len(variables)
# for n in range(1, generators+1):
#     for p in phases:
#         variables.append(f"Igr{p}")
#     for p in phases:
#         variables.append(f"Igi{p}")
# end = len(variables)
# positions['generators'] = (start, end)

# # Load currents
# start = len(variables)
# for n in range(1, loads+1):
#     for p in phases:
#         variables.append(f"Ilr{p}")
#     for p in phases:
#         variables.append(f"Ili{p}")
# end = len(variables)
# positions['loads'] = (start, end)

# #McCormick Variables
# start = len(variables)
# for n in range(1, loads +1):
#     for p in phases:
#         variables.append(f"xr{p}")
#     for p in phases:
#         variables.append(f"yr{p}")
#     for p in phases:
#         variables.append(f"zr{p}")
#     for p in phases:
#         variables.append(f"xi{p}")
#     for p in phases:
#         variables.append(f"yi{p}")
#     for p in phases:
#         variables.append(f"zi{p}")
#     for p in phases:
#         variables.append(f"v{p}")
#     for p in phases:
#         variables.append(f"w{p}")
# end = len(variables)
# positions['Mc_Vars'] = (start,end)
    
    
    
# Zline = np.array([
#  [0.4576+1.078j, 0.1559 +0.5017j, 0.1535+0.3849j],
#  [0.1559+0.5017j, 0.4666+1.0482j, 0.158+0.4236j],
#  [0.1535+0.3849j, 0.158+0.4236j, 0.4615+1.0651j]
# ])

# gen_one = Generator('Node 1 Generator', 1, 12470, 3)
# line_one_two = Line('Line 1 to 2', 1, 2, 2000/5280, 3, Zline)
# line_one_two = Line('Line 3 to 4', 3, 4, 2500/5280, 3, Zline)
# trans_two_three = Transformer('Transformer 2 to 3', 2, 3, 'Y-Y', 12470/4160, 6000000, 12470,0.01+0.06j)
# load_4 = Load('Load 4', 4, 1800000, [0.9,0.9,0.9])


# both_1 = [line for line in Line.all_lines if line.to_node == 1 and line.from_node == 1]
# print(positions['generators'][0])
# print(powerflow(num_nodes, variables, phases, positions))