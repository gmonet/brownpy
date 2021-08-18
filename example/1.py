# https://numba.pydata.org/numba-doc/dev/cuda/simulator.html
import os 
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"

from brownpy.gpu_sim import Universe
import brownpy.topology as Top

seed = 0

dt = int(1E6) #fs (1ns) - time steps
D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

# Geometry
L = 1E3 # A (100nm) - channel length
h = 1E2 # A (10nm)  - channel height
R = 1E4 # A (1um) - reservoir size

N= 2*1024

top = Top.ElasticChannel1(L=L, h=h, R=R)
u = Universe(N=N, top=top, D=D, dt=dt,
             output_path='simu', seed=seed)

u.run(1_000_000, seed=seed);