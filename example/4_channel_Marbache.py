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
    
    def compute_rolling_var(intervals_dt, DeltaN, insideN, file_path):
        
        df = pd.DataFrame(DeltaN)
        DeltaN_rv = []
        for interval in intervals_dt:
            DeltaN_rv.append(df.rolling(interval).var().mean()[0])
        DeltaN_rv = np.array(DeltaN_rv)

        insideN = N-(data['right']+data['left'])
        df = pd.DataFrame(insideN)
        insideN_rv = []
        for interval in intervals_dt:
            insideN_rv.append(df.rolling(interval).var().mean()[0])
        insideN_rv = np.array(insideN_rv)
        with h5py.File(file_path, 'a') as f:
            rungrp = f['run']['0']
            rungrp.create_dataset('intervals_dt', data = intervals_dt)
            rungrp.create_dataset('DeltaN_rv', data = DeltaN_rv)
            rungrp.create_dataset('insideN_rv', data = insideN_rv)

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

    # Timestep 
    dt = int(0.05*Hc**2/D)
    N = 32*1024
    print(f'dt = {dt:.2e} fs = {prefix(dt*1E-15)}s')

    Nsteps = int(1.5*1E6/0.05) * 10
    print(f"Number of steps : {Nsteps:.2e} = {prefix(dt*Nsteps*1E-15)}s")

    # Time interval for computing rolling variance
    intervals = np.logspace(-1,6,40) # in Hc²/D
    intervals_dt = np.int64(intervals*Hc**2/D/dt) # In timesteps unit

    threads = []
    # H_factors = H/Hc
    H_factors = [1, 2, 10, 25, 100]
    # ar_factor = Lc/Hc
    ar_factors = [0, 1, 2, 5, 10, 100]
    for H_factor in H_factors:
        for ar_factor in ar_factors:
            print('###############')
            print(f'{H_factor}_{ar_factor}')
            output_path=Path(f'./4/channel/2/{H_factor}_{ar_factor}.hdf5')
            if output_path.exists():
                continue
            
            H = H_factor*Hc # Distance between channel
            Lc = ar_factor*Hc # Channel length

            top = Top.ElasticChannel2(Hc=Hc, Lc=Lc, 
                                    H=H, L=L)
            u=Universe(N=N, top=top, D=D, dt=dt,
                    output_path=output_path,
                    overwrite=True)
            u.run(Nsteps)
            time.sleep(1)
            
            # data = u[0]
            # DeltaN = (data['right'] - data['left']).astype(np.int16)
            # insideN = N-(data['right']+data['left'])
            
            # thread = Thread(target=compute_rolling_var, 
            #                 args=(intervals_dt, DeltaN, insideN, output_path))
            # thread.start()
            
    # for thread in threads:
    #     thread.join()
    

    # p = Path('/home/invites/gmonet/brownpy/example/4/channel/')
    # for path in p.glob('*.hdf5'):
    #     with h5py.File(path, 'r') as f:
    #         array_out = np.stack((f['run/0/intervals_dt'][...], 
    #                             f['run/0/DeltaN_rv'][...],
    #                             f['run/0/insideN_rv'][...])).T

    #     np.savetxt(path.with_suffix('.out'), array_out, delimiter=', ',
    #             header='intervals_dt, DeltaN_rv, insideN_rv')