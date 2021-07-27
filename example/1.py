from brownpy.gpu_sim import Universe
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from tqdm.auto import tqdm


plt.style.use('dark_background')
dt = int(1E6) #fs (1ns) - time steps
D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

# Geometry
L = 1E3 # A (100nm) - channel length
h = 1E2 # A (10nm)  - channel height
R = 1E4 # A (1um) - reservoir size

N= 8*1024

u = Universe(N=N, L=L, h=h, R=R, D=D, dt=dt,
             output_path='simu.nc')

u.run3(1_000_000_000);