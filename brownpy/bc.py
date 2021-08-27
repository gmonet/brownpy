import math
from numba import jit

@jit
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