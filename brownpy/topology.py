
import math
import textwrap

import h5py
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from numba import jit, cuda
from numpy import array, float32, ndarray, uint, uint32
from numba.cuda.random import xoroshiro128p_uniform_float32

from brownpy import bc

dtype=float32

# Maximum bounce during one step
# It may happen that the elastic bounce of a particle lead it to another wall    
MAX_BOUNCE = 4

  
class Topology():
  def compile_check_region(regions):
    '''
    Create device CUDA function from defined regions
    Args:
      regions (list of dict): Each input must be dictionnary with, at least,
        a 'name' and 'def' keys.
    Returns:
      check_region (cuda device function)
    '''
    code_check_region = f'''
    @cuda.jit(device=True)
    def check_region(x:nb.types.float32, z:nb.types.float32,
                      inside:nb.types.Array, 
                      step:nb.types.uint64, 
                      internal_state:tuple) -> None:
      pass
    '''
    for i, region in enumerate(regions):
      code_check_region+=f'''
      if {region['def']}:
        cuda.atomic.add(inside, ({i}, step), 1)
      '''
    code_check_region = textwrap.dedent(code_check_region)
    return_dict={}
    exec(code_check_region, globals(), return_dict)
    return return_dict['check_region']

  def fill_geometry(self, N: uint):
    """Randomly fill the geometry
    """
    raise NotImplementedError

  def to_hdf5(self, geom_grp: h5py.Group):
    raise NotImplementedError

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    raise NotImplementedError

  def compute_boundary_condition(self, 
                                 x0:dtype, z0:dtype, 
                                 x1:dtype, z1:dtype,
                                 rng_states:array, 
                                 internal_state:tuple):
    raise NotImplementedError

  def check_region(self,
                   x:dtype, z:dtype,
                   inside:nb.types.Array, 
                   step:nb.types.uint64, 
                   internal_state:tuple) -> None:
    raise NotImplementedError

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    raise NotImplementedError
  
  def plot(self, ax=None):
    raise NotImplementedError

class Infinite(Topology):
  __version__='0.0.1'
  def __init__(self, **kwargs) -> None:
    """Jut inifinite space without any walls
    Args:
      None
    kwargs (advanced parameters):
      seed (int): set seed
    """
    regions = []
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = Infinite.__name__
    geom_grp['version'] = Infinite.__version__

    Rgrp = geom_grp.create_group("reservoir")
    Cgrp = geom_grp.create_group("pore")

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls()

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)
    r0 = rng.uniform((-1, -1), (1, 1), size=(N, 2))
    return r0.astype(dtype)

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

class Periodic(Topology):
  __version__='0.0.1'
  def __init__(self, L: dtype, **kwargs) -> None:
    """Jut periodic box without any walls

    ┌┈┈┈┈┈┈┈┈┈┈┈┈┈┐ ↑  
    ┊             ┊ │ 
    ┊             ┊ │
    ┊             ┊ │ L 
    ┊             ┊ │
    ┊             ┊ │
    └┈┈┈┈┈┈┈┈┈┈┈┈┈┘ 🡓
     ←-----------→
            L   
    
    ┊ : Periodic condition

    Args:
      L (float in A): Box size
    kwargs (advanced parameters):
      seed (int): set seed
    """
    self.L = L
    regions = []
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      # Periodic condition
      x1 = (L/2 + x1)%(L) - L/2
      z1 = (L/2 + z1)%(L) - L/2

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = Periodic.__name__
    geom_grp['version'] = Periodic.__version__
    Rgrp = geom_grp.create_group("reservoir")
    Rgrp.attrs['L'] = self.L

    Cgrp = geom_grp.create_group("pore")

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls(L=geom_grp['reservoir'].attrs['L'],
              )

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L = self.L

    r0 = rng.uniform((-L/2, -L/2), (L/2, L/2), size=(N, 2))
    return r0.astype(dtype)

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L = self.L

    border_kwargs = {'c': 'green', 'ls': '--'}

    # Draw geometry

    ax.plot([-L/2, -L/2, L/2,  L/2], 
            [-L/2,  L/2, L/2, -L/2], 
            **border_kwargs)

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

class ElasticPore1(Topology):
  __version__='0.0.3'
  def __init__(self, Lm: dtype, L: dtype, R: dtype, **kwargs) -> None:
    """Simple elastic Pore
    Geometry inspired from Marbach 2020
    J. Chem. Phys. 154, 171101 (2021); doi: 10.1063/5.0047380

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
    kwargs (advanced parameters):
      seed (int): set seed
    """
    self.Lm = Lm
    self.L = L
    self.R = R
    regions = [{'name':'left', 'def':'x<=0'}]
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      #Intersection with left wall
      X, Z = -L/2, 0
      NX, NZ = 1, 0
      if x1<X:
        x1 = x1+2*(X-x1)

      # Intersection with membrane
      X, Z = 0, 0
      NX, NZ = 1, 0
      if x0*x1<=0 and x1!=x0:
        t = (X-x0)/(x1-x0)
        zint = t*z1 + (1-t)*z0
        if math.fabs(zint)>R:
          x1 *= -1
                              
      #Intersection with right wall
      X, Z = +L/2, 0
      NX, NZ = 1, 0
      if x1>X:
        x1 = x1+2*(X-x1)
      
      z1 = (Lm + z1)%(2*Lm) - Lm

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticPore1.__name__
    geom_grp['version'] = ElasticPore1.__version__
    Rgrp = geom_grp.create_group("reservoir")
    Rgrp.attrs['Lm'] = self.L
    Rgrp.attrs['L'] = self.L
    Rgrp.attrs['bc_x'] = 'elastic'
    Rgrp.attrs['bc_z'] = 'periodic'
    Rgrp.attrs['bc_x_membrane'] = 'elastic'

    Cgrp = geom_grp.create_group("pore")
    Cgrp.attrs['R'] = self.R
    Cgrp.attrs['bc'] = 'elastic'

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls(Lm=geom_grp['reservoir'].attrs['Lm'],
              L=geom_grp['reservoir'].attrs['L'],
              R=geom_grp['pore'].attrs['R'])

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    Lm, L = self.Lm, self.L

    r0 = rng.uniform((-L/2, -Lm), (L/2, Lm), size=(N, 2))
    return r0.astype(dtype)

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, Lm, R = self.L, self.Lm, self.R

    border_kwargs = {'c': 'r'}

    # Draw geometry

    ax.plot([-L/2, -L/2], [-Lm, +Lm], **border_kwargs)
    ax.plot([ L/2,  L/2], [-Lm, +Lm], **border_kwargs)

    ax.plot([0, 0], [R, Lm], **border_kwargs)
    ax.plot([0, 0], [-R, -Lm], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

class ElasticChannel1(Topology):
  __version__='0.0.5'
  def __init__(self, L: dtype, h: dtype, R: dtype, **kwargs) -> None:
    """Create a new channel geometry

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
    kwargs (advanced parameters):
      seed (int): set seed
    """
    # TODO : Use Marbach notation ?
    self.L = L
    self.h = h
    self.R = R
    regions = [{'name':'left', 'def':'x<=0'}]
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2
        
        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1)<R+L/2) and (math.fabs(x1)>L/2) and \
           (math.fabs(x0)<R+L/2) and (math.fabs(x0)>L/2):
           break
        
        #Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1<X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0<X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with bottom channel
        X, Z = 0, -h/2
        NX, NZ = 0, 1
        if z0>Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                toCheck = True
                continue
        
        # Intersection with top channel
        X, Z = 0, +h/2
        NX, NZ = 0, 1
        if z0<Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                toCheck = True
                continue
                                
        # Intersection with right membrane
        X, Z = +L/2, 0
        NX, NZ = 1, 0
        if x0>X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with right wall
        X, Z = +R+L/2, 0
        NX, NZ = 1, 0
        if x1>X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break
        
        

        
        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)
      
      # Periodic boundary condition along z:
      z1 = (R + z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticChannel1.__name__
    geom_grp['version'] = ElasticChannel1.__version__
    Rgrp = geom_grp.create_group("reservoir")
    Rgrp.attrs['R'] = self.R
    Rgrp.attrs['bc_x'] = 'elastic'
    Rgrp.attrs['bc_z'] = 'periodic'
    Rgrp.attrs['bc_x_membrane'] = 'elastic'

    Cgrp = geom_grp.create_group("channel")
    Cgrp.attrs['L'], Cgrp.attrs['h'] = self.L, self.h
    Cgrp.attrs['bc'] = 'elastic'

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls(L=geom_grp['channel'].attrs['L'],
              h=geom_grp['channel'].attrs['h'],
              R=geom_grp['reservoir'].attrs['R'])

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L, h, R = self.L, self.h, self.R
    # Surface of reservoirs
    S_R = R**2
    # Surface of the channel
    S_c = h*L

    # Put particles in reservoirs
    N_R = int(np.ceil(N*S_R/(S_R+S_c)))
    r0_R = rng.uniform(-R, R, size=(N_R, 2))

    r0_R[np.where(r0_R[:, 0] < 0), 0] -= L/2
    r0_R[np.where(r0_R[:, 0] >= 0), 0] += L/2

    # Number of particles in channel
    N_c = N-N_R
    if N_c > 0:
      r0_c = np.stack((rng.uniform(-L/2, L/2, size=(N_c)),
                       rng.uniform(-h/2, h/2, size=(N_c)))).T
      r0 = np.concatenate((r0_R, r0_c))
    else:
      r0 = r0_R

    return r0.astype(dtype)

  def plot(self, ax=None, border_kwargs={'c': 'r'}):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    # Draw geometry

    ax.plot([-L/2-R, -L/2-R], [-R, +R], **border_kwargs)
    ax.plot([ L/2+R,  L/2+R], [-R, +R], **border_kwargs)

    ax.plot([-L/2, -L/2], [R, h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [R, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [h/2, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [-h/2, -h/2], **border_kwargs)
    ax.plot([-L/2, -L/2], [-R, -h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [-R, -h/2], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

class AbsorbingChannel1(Topology):
  __version__='0.0.4'
  def __init__(self, L: dtype, h: dtype, R: dtype, l: float, **kwargs) -> None:
    """Create a new channel geometry with absorbing wall

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
    kwargs (advanced parameters):
      seed (int): set seed
    """

    self.L = L
    self.h = h
    self.R = R
    self.l = l
    regions = [{'name':'left', 'def':'x<=0'},
               {'name':'absorbed', 'def':'internal_state[0] > 0'}]
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @cuda.jit(device=True)
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      pos = cuda.grid(1)
      toCheck = True
      i_BOUNCE = 0

      if internal_state[0] != 0:
        internal_state[0] -= 1
        x1 = x0
        z1 = z0
        return x1, z1
        
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2
        
        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1)<R+L/2) and (math.fabs(x1)>L/2) and \
           (math.fabs(x0)<R+L/2) and (math.fabs(x0)>L/2):
           break
        
        #Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1<X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0<X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with bottom channel
        X, Z = 0, -h/2
        NX, NZ = 0, 1
        if z0>Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = xint, zint+1
                T = -(1/l)*math.log(1-xoroshiro128p_uniform_float32(rng_states, pos))
                internal_state[0] = uint32(T)
                break
        
        # Intersection with top channel
        X, Z = 0, +h/2
        NX, NZ = 0, 1
        if z0<Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = xint, zint-1
                T = -(1/l)*math.log(1-xoroshiro128p_uniform_float32(rng_states, pos))
                internal_state[0] = uint32(T)
                break
                                
        # Intersection with right membrane
        X, Z = +L/2, 0
        NX, NZ = 1, 0
        if x0>X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with right wall
        X, Z = +R+L/2, 0
        NX, NZ = 1, 0
        if x1>X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break
        
        

        
        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)
      
      # Periodic boundary condition along z:
      z1 = (R + z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)
    
  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = AbsorbingChannel1.__name__
    geom_grp['version'] = AbsorbingChannel1.__version__
    Rgrp = geom_grp.create_group("reservoir")
    Rgrp.attrs['R'] = self.R
    Rgrp.attrs['bc_x'] = 'elastic'
    Rgrp.attrs['bc_z'] = 'periodic'
    Rgrp.attrs['bc_x_membrane'] = 'elastic'

    Cgrp = geom_grp.create_group("channel")
    Cgrp.attrs['L'], Cgrp.attrs['h'] = self.L, self.h
    Cgrp.attrs['l'] = self.l
    Cgrp.attrs['bc'] = 'absorbing'

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls(L=geom_grp['channel'].attrs['L'],
              h=geom_grp['channel'].attrs['h'],
              R=geom_grp['reservoir'].attrs['R'],
              l=geom_grp['channel'].attrs['l'])

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L, h, R = self.L, self.h, self.R
    # Surface of reservoirs
    S_R = R**2
    # Surface of the channel
    S_c = h*L

    # Put particles in reservoirs
    N_R = int(np.ceil(N*S_R/(S_R+S_c)))
    r0_R = rng.uniform(-R, R, size=(N_R, 2))

    r0_R[np.where(r0_R[:, 0] < 0), 0] -= L/2
    r0_R[np.where(r0_R[:, 0] >= 0), 0] += L/2

    # Number of particles in channel
    N_c = N-N_R
    if N_c > 0:
      r0_c = np.stack((rng.uniform(-L/2, L/2, size=(N_c)),
                       rng.uniform(-h/2, h/2, size=(N_c)))).T
      r0 = np.concatenate((r0_R, r0_c))
    else:
      r0 = r0_R

    return r0.astype(dtype)

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    border_kwargs = {'c': 'r'}
    border_abs_kwargs = {'c': 'b'}

    # Draw geometry

    ax.plot([-L/2-R, -L/2-R], [-R, +R], **border_kwargs)
    ax.plot([ L/2+R,  L/2+R], [-R, +R], **border_kwargs)

    ax.plot([-L/2, -L/2], [R, h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [R, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [h/2, h/2], **border_abs_kwargs)
    ax.plot([-L/2, +L/2], [-h/2, -h/2], **border_abs_kwargs)
    ax.plot([-L/2, -L/2], [-R, -h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [-R, -h/2], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

class SpeedElasticChannel1_dev(Topology):
  __version__='0.0.1'
  def __init__(self, L: dtype, h: dtype, R: dtype, **kwargs) -> None:
    """Create a new channel geometry

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
    kwargs (advanced parameters):
      seed (int): set seed
    """
    # TODO : Use Marbach notation ?
    self.L = L
    self.h = h
    self.R = R
    regions = []
    self.regions = regions

    ## Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2
        
        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1)<R+L/2) and (math.fabs(x1)>L/2) and \
           (math.fabs(x0)<R+L/2) and (math.fabs(x0)>L/2):
           break
        
        #Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1<X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0<X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with bottom channel
        X, Z = 0, -h/2
        NX, NZ = 0, 1
        if z0>Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                toCheck = True
                continue
        
        # Intersection with top channel
        X, Z = 0, +h/2
        NX, NZ = 0, 1
        if z0<Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                toCheck = True
                continue
                                
        # Intersection with right membrane
        X, Z = +L/2, 0
        NX, NZ = 1, 0
        if x0>X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den!=0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t>0) and (t<1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
                break

        # Intersection with right wall
        X, Z = +R+L/2, 0
        NX, NZ = 1, 0
        if x1>X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break
        
        

        
        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)
      
      # Periodic boundary condition along z:
      z1 = (R + z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    self.check_region = Topology.compile_check_region(regions)

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticChannel1.__name__
    geom_grp['version'] = ElasticChannel1.__version__
    Rgrp = geom_grp.create_group("reservoir")
    Rgrp.attrs['R'] = self.R
    Rgrp.attrs['bc_x'] = 'elastic'
    Rgrp.attrs['bc_z'] = 'periodic'
    Rgrp.attrs['bc_x_membrane'] = 'elastic'

    Cgrp = geom_grp.create_group("channel")
    Cgrp.attrs['L'], Cgrp.attrs['h'] = self.L, self.h
    Cgrp.attrs['bc'] = 'elastic'

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    top = cls(L=geom_grp['channel'].attrs['L'],
              h=geom_grp['channel'].attrs['h'],
              R=geom_grp['reservoir'].attrs['R'])

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L, h, R = self.L, self.h, self.R
    # Surface of reservoirs
    S_R = R**2
    # Surface of the channel
    S_c = h*L

    # Put particles in reservoirs
    N_R = int(np.ceil(N*S_R/(S_R+S_c)))
    r0_R = rng.uniform(-R, R, size=(N_R, 2))

    r0_R[np.where(r0_R[:, 0] < 0), 0] -= L/2
    r0_R[np.where(r0_R[:, 0] >= 0), 0] += L/2

    # Number of particles in channel
    N_c = N-N_R
    if N_c > 0:
      r0_c = np.stack((rng.uniform(-L/2, L/2, size=(N_c)),
                       rng.uniform(-h/2, h/2, size=(N_c)))).T
      r0 = np.concatenate((r0_R, r0_c))
    else:
      r0 = r0_R

    return r0.astype(dtype)

  def plot(self, ax=None, border_kwargs={'c': 'r'}):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    # Draw geometry

    ax.plot([-L/2-R, -L/2-R], [-R, +R], **border_kwargs)
    ax.plot([ L/2+R,  L/2+R], [-R, +R], **border_kwargs)

    ax.plot([-L/2, -L/2], [R, h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [R, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [h/2, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [-h/2, -h/2], **border_kwargs)
    ax.plot([-L/2, -L/2], [-R, -h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [-R, -h/2], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax

_topologyDic = {ElasticPore1.__name__:ElasticPore1,
                ElasticChannel1.__name__: ElasticChannel1,
                AbsorbingChannel1.__name__: AbsorbingChannel1}
