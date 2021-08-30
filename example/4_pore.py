from brownpy.gpu_sim import Universe
import brownpy.topology as Top
from brownpy.utils import prefix
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
import numpy as np
from tqdm.auto import tqdm
plt.style.use('dark_background')

if __name__ == "__main__":
    # Diffusion coefficient
    D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

    # Geometry
    R = 1E2 # A (10nm) - Pore radius
    L = 500*R # A - Reservoir depht
    factor = 0.10
    Lm = R/factor # A - Reservoir height

    # Timestep 
    dt_marbach = 0.05*R**2/D
    dt = int(10**np.floor(np.log10(dt_marbach)))
    print(f'dt = {dt:.2e} fs = {prefix(dt*1E-15)}s')
    N= 8*1024

    top = Top.ElasticPore1(L=L, Lm=Lm, R=R)
    u = Universe(N=N, top=top, D=D, dt=dt,
                output_path=f'pore_{factor:.2f}')

    Nsteps = int(1.5*1E6*R**2/D/dt)

    u.run(Nsteps//10);