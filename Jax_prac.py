import jax
import jax.numpy as jnp

def opf_eqs():
    
    return

def initFunc(InputVals):
    Vupper, Vlower, Iupper, PL, QL, sV, nt, ztbase, ztpu = InputVals 
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
    zt = ztpu * ztbase
    zphase = jnp.array([
        [zt, 0, 0],
        [0, zt, 0],
        [0, 0, zt]
    ])
    Yt = jnp.linalg.inv(zphase)
    Gtr = Yt.real
    Bti = Yt.imag
    Zline = jnp.array([
        [0.4576 + 1.078j, 0.1559 + 0.5017j, 0.1535 + 0.3849j],
        [0.1559 + 0.5017j, 0.4666 + 1.0482j, 0.158 + 0.4236j],
        [0.1535 + 0.3849j, 0.158 + 0.4236j, 0.4615 + 1.0651j]
    ])
    Yline12 = jnp.linalg.inv(Zline * (2000 / 5280))
    Yline34 = jnp.linalg.inv(Zline * (2000 / 5280))
    Gl12 = Yline12.real
    Bl12 = Yline12.imag
    Gl34 = Yline34.real
    Bl34 = Yline34.imag
    #V4r, V4i, w,v,z,z1
    upper = jnp.array([
        Vlower, Vlower, Vlower, Vlower, Vlower, Vlower,Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2,
        Iupper, Iupper, Iupper, Iupper, Iupper, Iupper]).reshape(-1, 1)
    lower = jnp.array([
        -Vlower, -Vlower, -Vlower, -Vlower, -Vlower, -Vlower,0, 0, 0, 0, 0, 0,
        -Iupper, -Iupper, -Iupper, -Iupper, -Iupper, -Iupper]).reshape(-1, 1)
#If we are going to be updating the upper and lower bounds, then we will need to update the A matrix for Ax-b<=0 each time...
    origUpper = upper.copy()
    origLower = lower.copy()

    obj = ['V4r', 'V4i', 'w', 'v', 'z0', 'z1']
    sense0 = ['minimize', 'maximize']
    phase = [0, 1, 2]

    return PL, QL, sV, Vupper, Vlower, Gl12, Bl12, Gtr, Bti, Gl34, Bl34, \
           Vsr, Vsi, nt, upper, lower, origLower, origUpper, obj, sense0, phase

# def eqCon1(Islackr,V1r,V1i,V2r,V2i): #Real and imaginary current at node 1
#     eqVec1 = []
#     for i in range(Islackr.shape[0]):
#         expr = Islackr[i] + sum(Gl12[i,j]*(V1r[j]-V2r[j]) for j in range(sV)) - sum(Bl12[i,j]*(V1i[j]-V2i[j]) for j in range(sV))
#         eqVec1.append(expr)  
#     return eqVec1
    
#Initialize 
#Vupper, VLower, Zupper, PL, QL, sV, nt, ztbase, ztpu
InitVal = (12470 / jnp.sqrt(3), 4160 / jnp.sqrt(3), 1200, 1800000, (1800000 / 0.9) * jnp.sin(jnp.arccos(0.9)), 3, 12470 / 4160, (4160**2) / 6000000, 0.01 + 0.06j)
(PL,QL,sV,Vupper,Vlower,Gl12,Bl12,Gtr,Bti,Gl34,Bl34,
     Vsr,Vsi,nt,upper,lower,origLower,origUpper,obj,sense0,phase) = initFunc(InitVal)
# test = eqCon1()

# arr = jnp.ones((4,2,3))
rows = jnp.zeros((14,3))
base = jnp.tile(rows, (3,1))
array = ['v1R','v1I','v2R', 'v2I', 'v3R', 'v3I', 'v4R', 'v4I', 
         'iSlackr', 'iSlacki', 'Ixr','Ixi', 'I2xr','I2xi','z0', 'z1']

names = {name: base.copy() for name in array}  

for i in range(rows.size):
    if i < 3:
        names['v1R'] = names['v1R'].at[i].set(Gl12[i,:])
        names['v1I'] = names['v1I'].at[i].set(-Bl12[i,:])
        names['v2R'] = names['v2R'].at[i].set(-Gl12[i,:])
        names['v2I'] = names['v2I'].at[i].set(Bl12[i,:])
        names['iSlackr'] = names['iSlackr'].at[i].set(-jnp.eye(3)[i,:])
    if 2 < i < 6:
        names['v1R'] = names['v1R'].at[i].set(Bl12[i-3,:])
        names['v1I'] = names['v1I'].at[i].set(Gl12[i-3,:])
        names['v2R'] = names['v2R'].at[i].set(-Bl12[i-3,:])
        names['v2I'] = names['v2I'].at[i].set(-Gl12[i-3,:])
        names['iSlacki'] = names['iSlacki'].at[i].set(-jnp.eye(3)[i-3,:])
    if 5 < i < 9:
        names['v1R'] = names['v1R'].at[i].set(-jnp.eye(3)[i-6,:])
    if 8 < i < 12:
        names['v1I'] = names['v1I'].at[i].set(-jnp.eye(3)[i-9,:])
    if 11 < i < 15:
        names['v1R'] = names['v1R'].at[i].set(-Gl12[i-12,:])
        names['v1I'] = names['v1I'].at[i].set(Bl12[i-12,:])
        names['v2R'] = names['v2R'].at[i].set(Gl12[i-12,:])
        names['v2I'] = names['v2I'].at[i].set(-Bl12[i-12,:])
        names['Ixr']=names['Ixr'].at[i].set(jnp.eye(3)[i-12,:])
    if 14 < i < 18:
        names['v1R'] = names['v1R'].at[i].set(-Bl12[i-15,:])
        names['v1I'] = names['v1I'].at[i].set(-Gl12[i-15,:])
        names['v2R'] = names['v2R'].at[i].set(Bl12[i-15,:])
        names['v2I'] = names['v2I'].at[i].set(Gl12[i-15,:])
        names['Ixi']=names['Ixi'].at[i].set(jnp.eye(3)[i-15,:])
    if 17 < i < 21:
        names['Ixr']=names['Ixr'].at[i].set(jnp.eye(3)[i-18,:])
        names['I2xr']=names['I2xr'].at[i].set(-jnp.eye(3)[i-18,:])
    if 20 < i < 24:
        names['Ixi'] = names['Ixi'].at[i].set(jnp.eye(3)[i-21,:])
        names['I2xi'] = names['I2xi'].at[i].set(-jnp.eye(3)[i-21,:])
    if 23 < i < 27:
        names['I2xr' ]= names['I2xr'].at[i].set(-jnp.eye(3)[i-24,:])
        names['v2R'] = names['v2R'].at[i].set((1/nt)*Gtr[i-24,:])
        names['v3R'] = names['v3R'].at[i].set(-Gtr[i-24,:])
        names['v2I'] = names['v2I'].at[i].set((1/nt)*Bti[i-24,:])
        names['v3I'] = names['v3I'].at[i].set(Bti[i-24,:])        
    if 26 < i < 30:
        names['I2xi']=names['I2xi'].at[i].set(-jnp.eye(3)[i-27,:])
        names['v2R'] = names['v2R'].at[i].set((1/nt)*Bti[i-27,:])
        names['v3R'] = names['v3R'].at[i].set(-Bti[i-27,:])
        names['v2I'] = names['v2I'].at[i].set((1/nt)*Gtr[i-27,:])
        names['v3I'] = names['v3I'].at[i].set(Gtr[i-27,:])       
    if 29 < i < 33:
        names['v3R']=names['v3R'].at[i].set(Gtr[i-30,:] + Gl34[i-30,:])
        names['v2R']=names['v2R'].at[i].set(-(1/nt)*Gtr[i-30,:])
        names['v3I']=names['v3I'].at[i].set(Bti[i-30,:] - Bl34[i-30,:])
        names['v2I']=names['v2I'].at[i].set((1/nt)*Bti[i-30,:])
        names['v4R']=names['v4R'].at[i].set(-Gl34[i-20,:])
        names['v4I']=names['v4I'].at[i].set(Bl34[i-30,:])
    if 32 < i < 36:
        names['v3I']=names['v3I'].at[i].set(Gtr[i-33,:]+Gl34[i-33,:])
        names['v2I']=names['v2I'].at[i].set(-(1/nt)*Gtr[i-33,:])
        names['v3R']=names['v3R'].at[i].set(Bti[i-33,:])
        names['v2R'] =names['v2R'].at[i].set((1/nt)*Bti[i-33,:])
        names['v4I']=names['v4I'].at[i].set(-Gl34[i-33,:])
        names['v4R']=names['v4R'].at[i].set(-Bl34[i-33,:])
    if 35 < i < 39:
        names['v4R']=names['v4R'].at[i].set(Gl34[i-36,:])
        names['v3R']=names['v3R'].at[i].set(-Gl34[i-36,:])
        names['v4I']=names['v4I'].at[i].set(-Bl34[i-36,:])
        names['v3I']=names['v3I'].at[i].set(Bl34[i-36,:])
        names['z0']=names['z0'].at[i].set(jnp.eye(3)[i-36,:])
    if 38 < i < 42:
        names['v4I']=names['v4I'].at[i].set(Gl34[i-39,:])
        names['v3I']=names['v3I'].at[i].set(-Gl34[i-39,i])
        names['v4R']=names['v4R'].at[i].set(Bl34[i-39,:])
        names['v3R']=names['v3R'].at[i].set(-Bl34[i-39,:])
        names['z1']=names['z1'].at[i].set(jnp.eye(3)[i-39,:])
# print(names['v4R'])

A1 = jnp.column_stack([names[key] for key in array])
print(A1.shape)
print(len(names))
numMCeq = (4*4) + (2*3)
#Create rows based on single phase representation of problem
rows1 = jnp.zeros((numMCeq,3))
#Increase to 3-phase
base1 = jnp.tile(rows1, (3,1))
bVec = jnp.zeros((base.shape[0],1))
#create an empty array for each variable
array1 = ['V4r', 'V4i','v', 'w', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1']
names1 = {name: base1.copy() for name in array1}
lil = jnp.eye(3)@upper[6:9,:]
        
for i in range(bVec.size):
    if i < 3:
        #x=zv (x,y) (equations 0-11)
        bVec = bVec.at[i].set(upper[i+12] * upper[i+6])        # XUYU
        bVec = bVec.at[i+3].set(-upper[i+12] * lower[i+6])     # XUYU
        bVec = bVec.at[i+6].set(-lower[i+12] * upper[i+6])
        bVec = bVec.at[i+9].set(lower[i+12] * lower[i+6])

        names1['z0'] = names1['z0'].at[i].set((jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
        names1['z0'] = names1['z0'].at[i+3].set(-(jnp.eye(3)@lower[6:9,:])[i,:])
        names1['z0'] = names1['z0'].at[i+6].set(-(jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
        names1['z0'] = names1['z0'].at[i+9].set(-(jnp.eye(3)@lower[6:9,:])[i,:])

        names1['v'] = names1['v'].at[i].set((jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+3].set(-(jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+6].set(-(jnp.eye(3)@lower[12:15,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+9].set((jnp.eye(3)@lower[12:15,:])[i,:]) #yXU

        names1['x0'] = names1['x0'].at[i].set(-jnp.eye(3)[i,:])
        names1['x0'] = names1['x0'].at[i+3].set(jnp.eye(3)[i,:])
        names1['x0'] = names1['x0'].at[i+6].set(jnp.eye(3)[i,:])
        names1['x0'] = names1['x0'].at[i+9].set(-jnp.eye(3)[i,:])
        # x1 = z1 v (equations 12-23)
        bVec = bVec.at[i+12].set(upper[i+15] * upper[i+6])
        bVec = bVec.at[i+15].set(-upper[i+15] * lower[i+6])
        bVec = bVec.at[i+18].set(-lower[i+15] * upper[i+6])
        bVec = bVec.at[i+21].set(lower[i+15] * lower[i+6])

        names1['z1'] = names1['z1'].at[i+12].set((jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
        names1['z1'] = names1['z1'].at[i+15].set(-(jnp.eye(3)@lower[6:9,:])[i,:])
        names1['z1'] = names1['z1'].at[i+18].set(-(jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
        names1['z1'] = names1['z1'].at[i+21].set(-(jnp.eye(3)@lower[6:9,:])[i,:])

        names1['v'] = names1['v'].at[i+12].set((jnp.eye(3)@upper[15:18,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+15].set(-(jnp.eye(3)@upper[15:18,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+18].set(-(jnp.eye(3)@lower[15:18,:])[i,:]) #yXU
        names1['v'] = names1['v'].at[i+21].set((jnp.eye(3)@lower[15:18,:])[i,:]) #yXU

        names1['x1'] = names1['x1'].at[i+12].set(-jnp.eye(3)[i,:])
        names1['x1'] = names1['x1'].at[i+15].set(jnp.eye(3)[i,:])
        names1['x1'] = names1['x1'].at[i+18].set(jnp.eye(3)[i,:])
        names1['x1'] = names1['x1'].at[i+21].set(-jnp.eye(3)[i,:])
        # y = z w (equations 24-33)
        bVec = bVec.at[i+24].set(upper[i+12] * upper[i+9])
        bVec = bVec.at[i+27].set(-upper[i+12] * lower[i+9])
        bVec = bVec.at[i+30].set(-lower[i+12] * upper[i+9])
        bVec = bVec.at[i+33].set(lower[i+12] * lower[i+9])
        
        names1['z0'] = names1['z0'].at[i+24].set((jnp.eye(3)@upper[9:12,:])[i,:]) #xYU
        names1['z0'] = names1['z0'].at[i+27].set(-(jnp.eye(3)@lower[9:12,:])[i,:])
        names1['z0'] = names1['z0'].at[i+30].set(-(jnp.eye(3)@upper[9:12,:])[i,:]) #xYU
        names1['z0'] = names1['z0'].at[i+33].set(-(jnp.eye(3)@lower[9:12,:])[i,:])

        names1['w'] = names1['w'].at[i+24].set((jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+27].set(-(jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+30].set(-(jnp.eye(3)@lower[12:15,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+33].set((jnp.eye(3)@lower[12:15,:])[i,:]) #yXU

        names1['y0'] = names1['y0'].at[i+24].set(-jnp.eye(3)[i,:])
        names1['y0'] = names1['y0'].at[i+27].set(jnp.eye(3)[i,:])
        names1['y0'] = names1['y0'].at[i+30].set(jnp.eye(3)[i,:])
        names1['y0'] = names1['y0'].at[i+33].set(-jnp.eye(3)[i,:])
        # y1 = z1 w (equations 36-45)
        bVec = bVec.at[i+36].set(upper[i+15] * upper[i+9])
        bVec = bVec.at[i+39].set(-upper[i+15] * lower[i+9])
        bVec = bVec.at[i+42].set(-lower[i+15] * upper[i+9])
        bVec = bVec.at[i+45].set(lower[i+15] * lower[i+9])

        names1['z1'] = names1['z1'].at[i+36].set((jnp.eye(3)@upper[9:12,:])[i,:]) #xYU
        names1['z1'] = names1['z1'].at[i+39].set(-(jnp.eye(3)@lower[9:12,:])[i,:])
        names1['z1'] = names1['z1'].at[i+42].set(-(jnp.eye(3)@upper[9:12,:])[i,:]) #xYU
        names1['z1'] = names1['z1'].at[i+45].set(-(jnp.eye(3)@lower[9:12,:])[i,:])

        names1['w'] = names1['w'].at[i+36].set((jnp.eye(3)@upper[15:18,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+39].set(-(jnp.eye(3)@upper[15:18,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+42].set(-(jnp.eye(3)@lower[15:18,:])[i,:]) #yXU
        names1['w'] = names1['w'].at[i+45].set((jnp.eye(3)@lower[15:18,:])[i,:]) #yXU

        names1['y1'] = names1['y1'].at[i+36].set(-jnp.eye(3)[i,:])
        names1['y1'] = names1['y1'].at[i+39].set(jnp.eye(3)[i,:])
        names1['y1'] = names1['y1'].at[i+42].set(jnp.eye(3)[i,:])
        names1['y1'] = names1['y1'].at[i+45].set(-jnp.eye(3)[i,:])
        
        # v = V4r^2 (equations 48-54)
        bVec = bVec.at[i+48].set(upper[i] * upper[i])
        bVec = bVec.at[i+51].set(lower[i] * lower[i])
        bVec = bVec.at[i+54].set(-lower[i] * upper[i])

        names1['v'] = names1['v'].at[i+48].set(-jnp.eye(3)[i,:])
        names1['v'] = names1['v'].at[i+51].set(-jnp.eye(3)[i,:])
        names1['v'] = names1['v'].at[i+54].set(jnp.eye(3)[i,:])

        names1['v4r'] = names1['v4r'].at[i+48].set(2*(jnp.eye(3)@upper[0:3])[i,:])
        names1['v4r'] = names1['v4r'].at[i+51].set(2*(jnp.eye(3)@lower[0:3,:])[i,:])
        names1['v4r'] = names1['v4r'].at[i+54].set(-(jnp.eye(3)@(upper[0:3,:]+lower[0:3,:]))[i,:])

       # w = V4i^2 (equations 57-63)
        bVec = bVec.at[i+57].set(upper[i+3] * upper[i+3])
        bVec = bVec.at[i+60].set(lower[i+3] * lower[i+3])
        bVec = bVec.at[i+63].set(-lower[i+3] * upper[i+3])

        names1['w'] = names1['w'].at[i+57].set(-jnp.eye(3)[i,:])
        names1['w'] = names1['w'].at[i+60].set(-jnp.eye(3)[i,:])
        names1['w'] = names1['w'].at[i+63].set(jnp.eye(3)[i,:])

        names1['v4i'] = names1['v4i'].at[i+57].set(2*(jnp.eye(3)@upper[3:6])[i,:])
        names1['v4i'] = names1['v4i'].at[i+60].set(2*(jnp.eye(3)@lower[3:6,:])[i,:])
        names1['v4i'] = names1['v4i'].at[i+63].set(-(jnp.eye(3)@(upper[3:6,:]+lower[3:6,:]))[i,:])


A2 = jnp.column_stack([names1[key1] for key1 in array1])
print(A2.shape)
print('####')
print(len(names))
    # if 2 < i < 6:
        
    # names['z'] = names['z'].at[i].set((jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
    # names['w'] = names['w'].at[i].set((jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
    # names['x'] = names['x'].at[i].set(-jnp.eye(3)[i,:])
    # if 2<i<6:
    #     bVec[i] = -upper[i+9]*lower[i+3]
    #     names['z'] = names['z'].at[i].set((jnp.eye(3)@upper[6:9,:])[i,:]) #xYU
    #     names['w'] = names['w'].at[i].set((jnp.eye(3)@upper[12:15,:])[i,:]) #yXU
    #     names['x'] = names['x'].at[i].set(-jnp.eye(3)[i,:])
"""
So I will need, for each phase, three of these
And these are all McCormick equations


def QuadMcCor(x, y, XU, XL):
    return[
        XU**2 >= 2*XU*x - y,
        XL**2 >= 2*x*XL - y,
        -XL*XU>= y - (XU+XL)*x
    ]
    

 #V4r, V4i, w,v,z,z1
    upper = jnp.array([
        V4r, V4i
        Vlower, Vlower, Vlower, Vlower, Vlower, Vlower,
        v,w
        Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2, Vlower**2,
        z0,z1
        Iupper, Iupper, Iupper, Iupper, Iupper, Iupper
    ]).reshape(-1, 1)
------
def McCormick(x, y, z, XU, XL, YU, YL):
    return[
        XU*YU >= YU*x + XU*y - z,
        -XU*YL >= z - YL*x - XU*y,
        -XL*YU >= z - YU*x - XL*y ,
        XL*YL >= YL*x + XL*y - z
    ]
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
"""
names = {name: base.copy() for name in array}  
for i in range(5):
    var = 'temp'

def equalF(x):
    return jnp.dot(A,x)
J=jax.jacobian(equalF)
print(J)

