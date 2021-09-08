import sys
import time
from io import StringIO
from pathlib import Path

import brownpy.topology as Top
import cpuinfo
import numba as nb
from brownpy.gpu_sim import Universe
from numba import cuda


#https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

if __name__ == '__main__':
    log = ''
    dt = int(1E6) #fs (1ns) - time steps
    D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

    # Geometry
    L = 1E3 # A (100nm) - channel length
    h = 1E2 # A (10nm)  - channel height
    R = 1E4 # A (1um) - reservoir size

    N = 4*1024

    top = Top.ElasticChannel1(L=L, h=h, R=R)
    u = Universe(N=N, top=top, D=D, dt=dt,
                output_path='bench', overwrite=True)
    log += u.f.__repr__() + '\n'

    
    N_steps = 1_000_000
    
    if cuda.is_available():
        log += '###################\n'
        log += 'GPU\n'
        with Capturing() as output:
            cuda.detect()
        log += '\n'.join(output) + '\n'
        u.run(1000, target='gpu') # warmup
        t0 = time.process_time_ns()
        u.run(N_steps, target='gpu')
        dt = time.perf_counter() - t0
        log += f'Speed: {dt/N/N_steps:.2f} ns/dt/p\n'

    log += '###################\n'
    log += 'CPU\n'
    log += f'{cpuinfo.get_cpu_info()["brand_raw"]}\n'
    log += f'Cores : {cpuinfo.get_cpu_info()["count"]}\n'
    u.run(1000, target='cpu') # warmup
    t0 = time.process_time_ns()
    u.run(N_steps, target='cpu')
    dt = time.perf_counter() - t0
    log += f'Speed: {dt/N/N_steps:.2f} ns/dt/p\n'
    with Path('./benchmark.log').open('w') as f:
        f.write(log)
