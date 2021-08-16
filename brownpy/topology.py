
import math
from abc import ABC, abstractmethod

import h5py
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from numba import cuda
from numpy import array, float32, ndarray, uint, uint32
from numba.cuda.random import xoroshiro128p_uniform_float32

from brownpy import bc

dtype=float32

# Maximum bounce during one step
# It may happen that the elastic bounce of a particle lead it to another wall    
MAX_BOUNCE = 4

class Topology(ABC):
  @abstractmethod
  def fill_geometry(self, N: uint):
    """Randomly fill the geometry
    """
    raise NotImplementedError

  @abstractmethod
  def to_hdf5(self, geom_grp: h5py.Group):
    raise NotImplementedError

  @classmethod
  @abstractmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    raise NotImplementedError

  @abstractmethod
  def compute_boundary_condition(self, 
                                 x0:dtype, z0:dtype, 
                                 x1:dtype, z1:dtype,
                                 rng_states:array, 
                                 internal_state:tuple):
    raise NotImplementedError

  @abstractmethod
  def check_region(self,
                   x:dtype, z:dtype,
                   inside:nb.types.Array, 
                   step:nb.types.uint64) -> None:
    raise NotImplementedError

  @abstractmethod
  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    raise NotImplementedError
  
  @abstractmethod
  def plot(self, ax=None):
    raise NotImplementedError

class ElasticChannelOld(Topology):
  def __init__(self, L: dtype, h: dtype, R: dtype, **kwargs) -> None:
    self.L = L
    self.h = h
    self.R = R
    
    ## Geometrical parameters are treated as constants during the compilation
    @cuda.jit(device=True, inline=True)
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      pos = cuda.grid(1)
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2
        # Left part
        if (x1 < -L/2):
          if (x1 < -R-L/2):
            x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                         x1, z1,
                                         -R-L/2, 0,
                                         1, 0)
            toCheck = True
          elif z1 > h/2 and x0 > -L/2:
            t = (z0-h/2)/(z0-z1)
            xint = t*x1 + (1-t)*x0
            zint = h/2
            if xint > -L/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, h/2,
                                           0, 1)
              toCheck = True
          elif z1 < -h/2 and x0 > -L/2:

            t = (z0+h/2)/(z0-z1)
            xint = t*x1 + (1-t)*x0
            zint = -h/2

            if xint > -L/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, -h/2,
                                           0, 1)
              toCheck = True
        # Right part
        elif (x1 > +L/2):
          if (x1 > +R+L/2):
            x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                         x1, z1,
                                         +R+L/2, 0,
                                         1, 0)
            toCheck = True
          elif z1 > h/2 and x0 < +L/2:
            t = (z0-h/2)/(z0-z1)
            xint = t*x1 + (1-t)*x0
            zint = h/2
            if xint < +L/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, h/2,
                                           0, 1)
              toCheck = True
          elif z1 < -h/2 and x0 < +L/2:
            t = (z0+h/2)/(z0-z1)
            xint = t*x1 + (1-t)*x0
            zint = -h/2
            if xint < +L/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, -h/2,
                                           0, 1)
              toCheck = True
        # Middle part
        else:
          if x0 < -L/2 and x1 > -L/2:
            t = (x0+L/2)/(x0-x1)
            xint = -L/2
            zint = t*z1 + (1-t)*z0
            if math.fabs(zint) > h/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           -L/2, 0,
                                           1, 0)
              toCheck = True
            else:
              if z1 > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             0, h/2,
                                             0, 1)
                toCheck = True
              elif z1 < -h/2:

                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             0, -h/2,
                                             0, 1)
                toCheck = True
          elif x0 > L/2 and x1 < L/2:
              # Intersection with x=+L/2
            t = (x0-L/2)/(x0-x1)
            xint = +L/2
            zint = t*z1 + (1-t)*z0
            if math.fabs(zint) > h/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           +L/2, 0,
                                           1, 0)
              toCheck = True
            else:
              if z1 > h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             0, h/2,
                                             0, 1)
                toCheck = True
              elif z1 < -h/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             0, -h/2,
                                             0, 1)
                toCheck = True
          else:  # x0 and x1 inside the channel
            if z1 > h/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, h/2,
                                           0, 1)
              toCheck = True
            elif z1 < -h/2:
              x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                           x1, z1,
                                           0, -h/2,
                                           0, 1)
              toCheck = True
        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)
      
      # Periodic boundary condition along z:
      z1 = (R - z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    @cuda.jit(device=True)
    def check_region(x:dtype, z:dtype,
                     inside:nb.types.Array, 
                     step:nb.types.uint64) -> None:
        if(x<=0):
          cuda.atomic.add(inside, (step), 1)

    self.check_region = check_region
  
  def compute_boundary_condition(self, 
                                 x0:dtype, z0:dtype, 
                                 x1:dtype, z1:dtype,
                                 rng_states:array, 
                                 internal_state:tuple):
    raise NotImplementedError

  def check_region(self,
                   x:dtype, z:dtype,
                   inside:nb.types.Array, 
                   step:nb.types.uint64) -> None:
    raise NotImplementedError

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticChannel1.__name__
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

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    border_kwargs = {'c': 'r'}

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

class ElasticChannel1(Topology):
  def __init__(self, L: dtype, h: dtype, R: dtype, **kwargs) -> None:
    self.L = L
    self.h = h
    self.R = R

    ## Geometrical parameters are treated as constants during the compilation
    @cuda.jit(device=True, inline=True)
    def compute_boundary_condition(x0:dtype, z0:dtype, 
                                   x1:dtype, z1:dtype,
                                   rng_states:array,
                                   internal_state:tuple):
      pos = cuda.grid(1)
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
      z1 = (R - z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    @cuda.jit(device=True)
    def check_region(x:dtype, z:dtype,
                     inside:nb.types.Array, 
                     step:nb.types.uint64) -> None:
        if(x<=0):
          cuda.atomic.add(inside, (step), 1)

    self.check_region = check_region
  
  def compute_boundary_condition(self, 
                                 x0:dtype, z0:dtype, 
                                 x1:dtype, z1:dtype,
                                 rng_states:array, 
                                 internal_state:tuple):
    raise NotImplementedError

  def check_region(self,
                   x:dtype, z:dtype,
                   inside:nb.types.Array, 
                   step:nb.types.uint64) -> None:
    raise NotImplementedError

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticChannel1.__name__
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

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    border_kwargs = {'c': 'r'}

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
  def __init__(self, L: dtype, h: dtype, R: dtype, l: float, **kwargs) -> None:
    self.L = L
    self.h = h
    self.R = R
    self.l = l

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
      z1 = (R - z1)%(2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    @cuda.jit(device=True)
    def check_region(x:dtype, z:dtype,
                     inside:nb.types.Array, 
                     step:nb.types.uint64) -> None:
        if(x<=0):
          cuda.atomic.add(inside, (step), 1)

    self.check_region = check_region
  
  def compute_boundary_condition(self, 
                                 x0:dtype, z0:dtype, 
                                 x1:dtype, z1:dtype,
                                 rng_states:array, 
                                 internal_state:tuple):
    raise NotImplementedError

  def check_region(self,
                   x:dtype, z:dtype,
                   inside:nb.types.Array, 
                   step:nb.types.uint64) -> None:
    raise NotImplementedError

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp['name'] = ElasticChannel1.__name__
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

  def plot(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h, R = self.L, self.h, self.R

    border_kwargs = {'c': 'r'}

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

_topologyDic = {ElasticChannel1.__name__: ElasticChannel1,
                ElasticChannelOld.__name__: ElasticChannelOld,
                AbsorbingChannel1.__name__: AbsorbingChannel1}
