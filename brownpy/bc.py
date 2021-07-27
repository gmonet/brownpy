import math
from numba import cuda
import numba as nb
# @cuda.jit('UniTuple(float32, 2)(float32,float32,float32,float32,float32,float32,float32,float32)', device=True, inline=True)
@cuda.jit(device=True,  inline=True)
def ReflectIntoPlane(x0,z0,
                     x1,z1,
                     X, Z, NX, NZ):
                     
    t = (NX*(x0-X) + NZ*(z0-Z))/(NX*(x0-x1) + NZ*(z0-z1))
    x_int = x1*t + x0*(1-t)
    z_int = z1*t + z0*(1-t)
    
    # Finding reflection
    x_1_int, z_1_int = x1-x_int, z1-z_int
    n_1_int = math.sqrt((x_1_int)**2 + (z_1_int)**2)
    ux_1_int, uz_1_int = x_1_int/n_1_int, z_1_int/n_1_int
    ps = (ux_1_int*NX + uz_1_int*NZ) # scalar product between n and u_1_int

    x1p = x_int + n_1_int*(ux_1_int - 2*ps*NX)
    z1p = z_int + n_1_int*(uz_1_int - 2*ps*NZ)
    return x1p, z1p

@cuda.jit('void(float32,float32,float32,float32,float32,float32,float32,float32)', device=True, inline=True)
def ReflectIntoPlane_dev(x0,z0,
                     x1,z1,
                     X, Z, NX, NZ):
    # print('WTF ! ')
    inter = NX*x0
    # t=math.(x0)
    # t=1/(NX*(x0))
    pass
    # t = (NX*(x0-X) + NZ*(z0-Z))/(NX*(x0-x1) + NZ*(z0-z1))
    # x_int = x1*t + x0*(1-t)
    # z_int = z1*t + z0*(1-t)
    
    # Finding reflection
    # x_1_int, z_1_int = x1-x_int, z1-z_int
    # n_1_int = math.sqrt((x_1_int)**2 + (z_1_int)**2)
    # ux_1_int, uz_1_int = x_1_int/n_1_int, z_1_int/n_1_int
    # ps = (ux_1_int*NX + uz_1_int*NZ) # scalar product between n and u_1_int

    # x1p = x_int + n_1_int*(ux_1_int - 2*ps*NX)
    # z1p = z_int + n_1_int*(uz_1_int - 2*ps*NZ)
    # return x1p, z1p

# @cuda.jit('UniTuple(float32, 2)(float32,float32,float32,float32,float32,float32,float32,float32,int32)', device=True)
# def ReflectIntoPlane_abs(x0,z0,
#                      x1,z1,
#                      X, Z, NX, NZ,
#                      t):
#     if t>0:
#         t-=1
#         return x0, z0
#     else:
#         return ReflectIntoPlane(x0,z0,
#                                 x1,z1,
#                                 X, Z, NX, NZ)

# @cuda.jit('UniTuple(float32, 2)(float32,float32,float32,float32,float32,float32,float32)', device=True)
@cuda.jit(device=True,  inline=True)
def ReflectIntoCircleSimplify(x0,z0,
                              x1,z1,
                              X,Z,R,):
    n1 = math.sqrt((x1-X)**2+(z1-Z)**2)
    ux_1_O, uz_1_O = (x1-X)/n1, (z1-Z)/n1
    x1p = X+0.99*(2*R-n1)*ux_1_O
    z1p = Z+0.99*(2*R-n1)*uz_1_O

    return x1p, z1p