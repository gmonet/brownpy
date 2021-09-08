from brownpy.settings import set_computation_type
set_computation_type('cpu')
from brownpy import gpu_sim
from brownpy.gpu_sim import Universe
import brownpy.topology as Top
from brownpy.utils import prefix, unwrap
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
import numpy as np
from threading import Thread
from pathlib import Path
import time
import pandas as pd
from tqdm.notebook import tqdm
plt.style.use('default')
import cupy as cp

if __name__ == "__main__":
    # Diffusion coefficient
    D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

    """
    ┃         ┃   ┃         ┃      ↑  
    ┃         ┃   ┃         ┃      │
    ┃         ┗━━━┛         ┃      │
    ┃                       ┃ ↕ Hc │ 2 H
    ┃         ┏━━━┓         ┃      │
    ┃         ┃   ┃         ┃      │
    ┃         ┃   ┃         ┃      ↓
    ←-------→ ←-→ ←-------→
        L      Lc     L
    """
    # Conversion into my notation
    Hc = 1E2 # A (10nm) - Channel width
    L = 250*Hc # Channel depth
    H_factor = 1
    H = H_factor*Hc # Distance between channel
    ar_factor = 100
    Lc = ar_factor*Hc # Channel length

    # Timestep 
    dt = int(0.05*Hc**2/D)
    N = 2*1024
    print(f'dt = {dt:.2e} fs = {prefix(dt*1E-15)}s')

    Nsteps = int(1.5*1E6/0.05) 
    print(f"Number of steps : {Nsteps:.2e} = {prefix(dt*Nsteps*1E-15)}s")


    top = Top.ElasticChannel2(Hc=Hc, Lc=Lc, 
                            H=H, L=L)
    u=Universe(N=N, top=top, D=D, dt=dt,
            output_path=f'./4/channel/{H_factor}_{ar_factor}',
            overwrite=True)

    u.run(1000, freq_dumps=100)