[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 

# brownpy: simulate brownian dynamics with Python and your GPU !
brownpy is a Python library that allows you to make 2D Brownian dynamics simulations on your Nvidia graphics card !

**Flexible**:
You can define your own simulation geometry in addition to the predefined geometries in brownpy.topology. Define your boundary conditions, the position of the walls and the physics of interaction between particles and walls (elastic scattering, absorption dynamics, etc...).

**High performance**:
Being [numba](https://github.com/numba/numba) based, brownpy allows you to run your simulations on all the cores of your processor or, even better, on your graphics card. A top generation nvidia graphics card can simulate 1 million time steps for 1000 particles in less than a second ! 

## Installation
Clone repository and install the package with pip:
```bash
git clone https://github.com/gmonet/brownpy.git
cd brownpy
pip install .
```
For brownpy to work with your graphics card, you must install the CUDA toolkit:
```bash
conda install cudatoolkit
```
For more details, see the [numba installation guide](https://numba.pydata.org/numba-doc/dev/user/installing.html).
## Examples
Take a look at the "example" folder, you will find:
1. [Here](./example/1_empty.ipynb) - A simulation of 10000 particles in an infinite simulation box. You will learn how to record and display the trajectory of the particles.
2. [Here](./example/2_pore.ipynb) - A simulation with two separate reservoirs with a membrane having a nanopore. You will see how to record and show the flow through this pore.

## Quick Start

#### Define the simulation setup.
```Python
from brownpy import Universe

# Define a geometry for your simulation (called topology) or use a predefined one 
from brownpy.topology import ElasticPore1
# Geometry inspired from Marbach 2020
# J. Chem. Phys. 154, 171101 (2021); doi: 10.1063/5.0047380

#     ┃         ┃         ┃     ↑  
#     ┃         ┃         ┃     │ Lm
#     ┃         ┃         ┃     │
#     ┃           ↕2R     ┃     ╵ 
#     ┃         ┃         ┃     
#     ┃         ┃         ┃     
#     ┃         ┃         ┃     
#      ←-----------------→
#               L   

N_particles = 8*1024 # Number of particles (power of 2 is better for gpu)
D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient
dt = 1e6 # fs (1ns) - Time step
seed = 1789 # Specify seed for reproductibility. The same seed is used
# for filling randomly the geometry and for random walk dynamic.

# Define topology
R = 10 # A (1nm) - Height of the pore
Lm = 1000 # A (100nm) - Distance between each pores
L = 1E4 # A (1um) - Reservoir depth 
top = ElasticPore1(Lm, L, R, seed=seed)

# Define simulation setup
u = Universe(top, N_particles, D, dt,
             output_path='elasticPore', overwrite=True)
```

#### Run a first short simulation to retrieve the trajectory of particles
```Python
u.run(10_000, freq_dumps=10) # Dump the position of particles each 10 steps.
# Carefull ! Beware, this has a significant impact on the performace 
# and can result in very heavy output files. For example this run will
# already generate 68MB of data !
# Read trajectory from hdf5 output file as a numpy array
traj = u.f['/run/0/trajectory']
```

#### Run a first Long time simulation to display the flux through the nanopore.
```Python
u.run(1_000_000) # Long time simulation to observe the flow through a nanopore.
left = u.f['/run/1/regions/left'] # Read the number of particle in the 
# left reservoir as a numpy array
```

### Predefined topology
As you have seen in the previous section, there are some predefined topologies. Here you will find more. Feel free to use their implementation in [`topology.py`](./brownpy/topology.py) to define your own topology (see next section).

```python
from brownpy import *
```
1. `Infinite`: Just an infinite space without any walls.
2. `Periodic`: Just periodic box without any walls.
3. `ElasticPore1`: Geometry inspired from Marbach 2020 (J. Chem. Phys. 154, 171101 (2021); doi: 10.1063/5.0047380)
```
        ┃         ┃         ┃     ↑  
        ┃         ┃         ┃     │ Lm
        ┃         ┃         ┃     │
        ┃                   ┃ ↕2R ╵ 
        ┃         ┃         ┃     
        ┃         ┃         ┃     
        ┃         ┃         ┃     
        ←-----------------→
                  L 

        ┃ : Elastic wall

    Args:
      Lm (float in A): Reservoir height
      L  (float in A): Reservoir depth
      R (float in A): Pore radius
```
4. `ElasticChannel1`: A channel geometry with elastic wall.
```
        ┃         ┃   ┃         ┃     ↑  
        ┃         ┃   ┃         ┃     │
        ┃         ┗━━━┛         ┃     │
        ┃                       ┃ ↕ h │ 2R
        ┃         ┏━━━┓         ┃     │
        ┃         ┃   ┃         ┃     │
        ┃         ┃   ┃         ┃     ↓
         ←-------→ ←-→ ←-------→
            R       L     R  

        ┃ : Elastic wall
    Args:
      L (float in A): Length of the channel
      h (float in A): Height of the channel
      R (float in A): Radius of reservoirs
```
5. `AbsorbingChannel1`: A channel geometry with elastic membrane and absorbing inner wall in the channel. 
```
        ┃         ┃   ┃         ┃     ↑  
        ┃         ┃   ┃         ┃     │
        ┃         ┗═══┛         ┃     │
        ┃                       ┃ ↕ h │ 2R
        ┃         ┏═══┓         ┃     │
        ┃         ┃   ┃         ┃     │
        ┃         ┃   ┃         ┃     ↓
         ←-------→ ←-→ ←-------→
            R       L     R

        ┃ : Elastic wall
        ═ : Absorbing wall
    Args:
      L (float in A): Length of the channel
      h (float in A): Height of the channel
      R (float in A): Radius of reservoirs
      l (float in dt-1) : Desorption frequency 
```

### User-defined topology
To define your own topology, you must create a child class of `Topology`. The structure of such a class follows the following scheme:

```python
from numba import jit, cuda
from brownpy import Topology
from brownpy.geometry import vector2D # Some operation on 2D vectors if needed. 

class myTopology(Topology):
    __version__ = '0.0.1'

    def __init__(self, myGeometryParameter:float32, **kwargs) -> None:
        self.myGeometryParameter = myGeometryParameter

        # The algorithm will count the number of particles within 
        # each region defined here. 
        regions = [{'name': 'myRegion', 'def': 'x<=1 and math.fabs(z)>=myGeometryParameter'},
                   {'name':'absorbed',  'def': 'internal_state[0] > 0'}] 
        # in 'def' create a python function with (x,y) as particule 
        # coordinates. You have also access to topology parameters
        # (here myGeometryParameter), math function and internal_state
        # array. Note: for now, internal_state can only store 1 float.
        self.regions = regions

        @jit
        def compute_boundary_condition(x0: dtype, z0: dtype,
                                       x1: dtype, z1: dtype,
                                       rng_states,
                                       internal_state: tuple):
        '''
        Args:
            x0, z0 (float): Coordinate of the particle at step t-1.
            x1, z1 (float): Coordinate of the particle at step t.
            rng_states (function): Random number generator. 
               See AbsorbingChannel1 for example.
            internal_state (tuple): internal custom attribute for a particule. 
              It can be the absorption time for example. 
              See AbsorbingChannel1 for example.
        '''
            # Here define the behavior of a particle crossing a wall of 
            # the limit of the simulation box.

            # For example: Elastic diffusion with vertical wall at X = -L/2
            X, Z = -L/2, 0
            NX, NZ = 1, 0
            if x1 < X:
                x1 = x1+2*(X-x1)

            # For example: a periodic boundary condition along z:
            z1 = (R + z1) % (2*R) - R # must define R in the __init__.

            return x1, z1
        self.compute_boundary_condition = compute_boundary_condition

        super().__init__()

    def fill_geometry(self, N: uint, seed=None) -> ndarray:
        # Define here how you want to fill your geometry before the simulation.
        # For example : randomly filling a square
        rng = np.random.default_rng(seed)
        L = self.L
        r0 = rng.uniform((-L/2, -L/2), (L/2, L/2), size=(N, 2))

        return r0.astype(float32)
    
    def plot(self, ax=None):
        # Define how you want to plot your geometry.
        return fig, ax

```

## Future
- Allow the user to add any piece of jitted python code before and/or after a distribution step in the main physics engine. This would give the possibility to add an interaction with an external forcefield. 
- Make `internal_state` extensible.
- Simpliy the part with random number generator.