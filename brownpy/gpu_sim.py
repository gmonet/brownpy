import math
import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from netCDF4 import Dataset
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float32)
from tqdm.auto import tqdm

from brownpy import bc
from brownpy.utils import prefix

# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


class Universe():
  """The Universe contains all the information describing the system.

  It contains information about the topologt of the system as the
  geometry, the timestep, the number of particles, their diffusion coefficient,
  the boundary condition ...

  Properties created with the ``@property`` decorator should be documented
  in the property's getter method.

  Attributes:
    attr1 (str): Description of `attr1`.
    attr2 (:obj:`int`, optional): Description of `attr2`.

  """
  __version__ = '0.0.2a'
  MAX_BOUNCE = 10

  def __init__(self, N: int,
                L: float, h: float, R: float, 
                D: float, dt: int,
                output_path: str,
                **kwargs):
    """Create a raw new Universe from topology parameters

    Args:
      N (int): Number of particles
      L (float in A): Length of the channel
      h (float in A): Height of the channel
      R (float in A): Radius of reservoirs
      D (float in A^2/fs): Diffusion coefficient
      dt (float in fs): Simulation timestep
      output_path (str): Output netcdf file
    kwargs (advanced parameters):
      gpu_memory (int, optional): GPU memory size to use. Default to 2*1024**3 (2GiB)
      threadsperblock (int, optional): Number of thread per block. Default to 128
      seed (int): set seed
    """
    self._mempool = cp.get_default_memory_pool()
    self._pinned_mempool = cp.get_default_pinned_memory_pool()

    # Define timestep
    self._dt = dt
    self._step = 0  # current step

    # Define geometry
    self._L, self._h, self._R = L, h, R

    # Define number of particles and their properties
    self._N = N
    self._D = D

    # Fix computation precision (TODO: make it compatible with saving in netcdf file)
    self._dtype = 'float32'

    # Fill randomly the geometry
    self._pos = cp.asarray(self._fillGeometry())

    # Cut the time axis into batch of gpu-computation in order to fill the gpu memory
    # 2GiB in default #TODO : not properly handled
    gpu_memory = kwargs.get('gpu_memory', 2*1024**3)
    self._batchSize = int(gpu_memory/4/2/N)  # Batch size in timestep

    # Thread per block (128 by default)
    self._threadsperblock = kwargs.get('threadsperblock', 128)

    # Set seed if defined by user or use the system clock as seed
    self._initial_seed = kwargs.get('seed', time.time_ns())
    np.random.seed(self._initial_seed%(2**32-1))
    cp.random.seed(self._initial_seed%(2**32-1))

    # Compile physical engine
    self._compile()

    self._output_path = output_path
    if not kwargs.get('_outputFileProvided', False):
      # Create a netcdf simulation file
      self._initOutputFile(output_path)
  
  @classmethod
  def from_nc(cls, input_path: str):
    """Create Universe from netcdf file previously created

    Args:
      input_path (str): Netcdf file
    """

    input_path = Path(input_path).resolve()
    if not input_path.exists():
      raise FileNotFoundError(f"{str(input_path)} doesn't exist.")

    with Dataset(str(input_path), "r", format="NETCDF4") as rootgrp:
      u = cls(N=rootgrp['particles'].N,
              L=rootgrp['geometry']['channel'].L,
              h=rootgrp['geometry']['channel'].h,
              R=rootgrp['geometry']['left_reservoir'].R,
              D=rootgrp['particles']['1'].D,
              dt=rootgrp.dt,
              output_path=input_path,
              _outputFileProvided=True)
    return u

  #region Properties
  @property
  def pos(self):
    """array: current position of particles (in A)."""
    return self._pos.get()

  @property
  def N(self):
    """int: Number of particles in the universe."""
    return self._N

  @property
  def L(self):
    """float: Length of the channel (in A)."""
    return self._L

  @property
  def h(self):
    """float: height of the channel (in A)."""
    return self._h

  @property
  def R(self):
    """float: Radius of reservoirs (in A)."""
    return self._R

  @property
  def D(self):
    """float: Diffusion coefficient (in A^2/fs)."""
    return self._D

  @property
  def dt(self):
    """float: Simulation timestep (in fs)."""
    return self._dt

  @property
  def precision(self):
    """type: Computation precision."""
    return self._dtype

  @property
  def batchSize(self):
    """int: Size of each batch along time axis used to complete GPU memory."""
    return self._batchSize

  @property
  def output_path(self):
    """str: Get path to the netcdf output path."""
    return self._output_path
  #endregion

  def __len__(self):
    with Dataset(self._output_path, "r", format="NETCDF4") as rootgrp:
      N_runs = len(rootgrp['run/'].groups)
    return N_runs

  def __getitem__(self, key):
    data = {}

    with Dataset(self._output_path, "r", format="NETCDF4") as rootgrp:
      groups_keys = list(rootgrp['run/'].groups.keys())

    if isinstance(key, int):
      key = str(key)

    if isinstance(key, str):
      split = 1000
      if key not in groups_keys:
        raise KeyError(f"Available runs : {', '.join(groups_keys)}")
      with Dataset(self._output_path, "r", format="NETCDF4") as rootgrp:
        for key, value in rootgrp[f'run/{key}'].variables.items():
          print(f'Reading {key} ...')
          # chunk is too slow !
          # data[key] = np.empty(shape = value.shape, 
          #                      dtype = value.dtype)
          # if value.ndim==1: # region variable
          #   total = value.shape[0]
          #   pbar = tqdm(total=total)
          #   for i in range(0, total, split):
          #     value[i:i+split]
          data[key] = value[:]
          print(f'... Done')
      return data
    else:
      raise TypeError(f'universe indices must be integers or str, not {type(key)}')
    

  def _initOutputFile(self, output_path: str):
    """Create and initialize the netcdf simulation file
    Args:
      output_path (str): Output netcdf file
    """
    # Check if the file already exists.
    # If yes, the file name will be incremented.
    output_path = Path(output_path).resolve()
    if output_path.exists():
        i = 1
        output_path_inc = output_path.with_name(
            output_path.stem+f'_{i}'+output_path.suffix)
        while output_path_inc.exists():
            i += 1
            output_path_inc = output_path.with_name(
                output_path.stem+f'_{i}'+output_path.suffix)
        print(
            f'{output_path.name} already exists, change output filename for {output_path_inc.name}')
        output_path = output_path_inc

    # Store output path name as attribut
    self._output_path = str(output_path)

    # # Create NETCDF4 file
    rootgrp = Dataset(self.output_path,
                      "w", format="NETCDF4")
    # Some general infos stored as attributes
    rootgrp.source = 'Created with BM_cuda'
    rootgrp.version = self.__version__
    rootgrp.date = "Created " + time.ctime(time.time())
    rootgrp.units = 'real'  # We use LAMMPS real units
    rootgrp.dt = self.dt
    rootgrp.dtype = self._dtype

    # Specify that it is a 2D simulation
    dim = rootgrp.createDimension("dim", 2)
    # Dimension idexing all particles
    rootgrp.createDimension("idx", self.N)
    # Create a group that will store information on particles
    # Like a topology folder
    # It will make sens if we have 2 types of particles (A and B) with
    # opposing charge for example.

    particlesgrp = rootgrp.createGroup("particles")
    particlesgrp.N = self.N

    pos = particlesgrp.createVariable("initial_pos", "f4", ("idx", "dim"))
    pos[:, :] = self.pos

    pos = particlesgrp.createVariable("current_pos", "f4", ("idx", "dim"))
    pos[:, :] = self.pos

    particlegrp_0 = particlesgrp.createGroup("1")
    particlegrp_0.type = 'A'
    particlegrp_0.D = self.D
    particlegrp_0.N = self.N

    # Create a group holding informations about geometry
    geomgrp = rootgrp.createGroup("geometry")
    LRgrp = geomgrp.createGroup("left_reservoir")
    LRgrp.R = self.R
    LRgrp.bc = 'elastic simplified'

    RRgrp = geomgrp.createGroup("right_reservoir")
    RRgrp.R = self.R
    RRgrp.bc = 'elastic simplified'

    Cgrp = geomgrp.createGroup("channel")
    Cgrp.L, Cgrp.h = self.L, self.h
    Cgrp.bc = 'elastic'

    # Create a group that will hold informations about simulations
    rungrp = rootgrp.createGroup("run")
    rungrp.run_number = 0  # number of simulation perfomed until now

    # Create a group for other paramaters (like the seed)
    sysgrp = rootgrp.createGroup("_sys")
    sysgrp.initial_seed = self._initial_seed

    rootgrp.close()

  def _fillGeometry(self):
    """Randomly fill the geometry
    """
    # Get geometry parameters
    L, h, R = self.L, self.h, self.R
    # Get number of particles
    N = self.N
    # Surface of reservoirs
    S_R = np.pi*R**2
    # Surface of the channel
    S_c = h*L

    # Put particles in reservoirs
    N_R = int(np.ceil(N*S_R/(S_R+S_c)))
    r0_R = np.random.uniform(-R, R, size=(N_R, 2))
    index_outside = np.where((np.linalg.norm(r0_R, axis=1, ord=2) > R) +
                              (np.linalg.norm(r0_R, axis=1, ord=2) < 10000))
    while(len(index_outside[0]) != 0):
      r0_R[index_outside] = np.random.uniform(
          -R, R, size=(len(index_outside[0]), 2))
      index_outside = np.where(np.linalg.norm(r0_R, axis=1, ord=2) > R)
    r0_R[np.where(r0_R[:, 0] < 0), 0] -= L/2
    r0_R[np.where(r0_R[:, 0] >= 0), 0] += L/2

    # Number of particles in channel
    N_c = N-N_R
    if N_c > 0:
      r0_c = np.stack((np.random.uniform(-L/2, L/2, size=(N_c)),
                        np.random.uniform(-h/2, h/2, size=(N_c)))).T
      r0 = np.concatenate((r0_R, r0_c))
    else:
      r0 = r0_R

    return r0.astype(self._dtype)

  def _compile(self):
    """Compile cuda function
    """
    ## Following parameters are treated as constants during the compilation
    ## It is faster than putting them as parameters for engine function #TODO IDK why
    # Get geometry parameters
    L, h, R = self.L, self.h, self.R
    # Maximum bounce during one step
    # It may happen that the elastic bounce of a particle lead it to another wall 
    MAX_BOUNCE = self.MAX_BOUNCE
    MAX_BOUNCE = 1
    # Get number of thread per block and compute the number of block necessary to 
    # account for all particles
    threadsperblock = self._threadsperblock
    if self.N % threadsperblock != 0:
      RuntimeWarning(
        f"""The number of particles should be a multiple of {threadsperblock} 
        for optimal performance""") #TODO : check if everyting work in this case
    blockspergrid = math.ceil(self.N / threadsperblock)
    dtype = self._dtype

    @cuda.jit(f'void({dtype}[:,:], {dtype}[:,:,:], uint32[:,:,:], {dtype}[:,:,:], uint32)')
    def engine_dump(r, dr, inside, history, step0):
      """Physical engine function compiled in CUDA to simulate pure brownian
      motion particles.

      Args:
        r (float32[:,:]): Initial position of particles
        dr (float32[:,:,:]): Displacement normaly generated with cupy (way faster than numba for now
                           see https://numba.discourse.group/t/random-array-generation-numba-cuda-slower-than-cupy/815)
        inside (uint32[:,:,:]): ouput array that will store the number of particles in predefinned regions as function of time 
        history (float32[:,:,:]): output array that wil store trajectory every custom timestep
        step0 (uint32): inital timestep
      """
      freq_dumps = math.ceil(dr.shape[2]/history.shape[2])
      pos = cuda.grid(1)
      if pos < r.shape[0]:
        i_DUMP = 0
        for step in range(dr.shape[2]):
          x0, z0 = r[pos, 0], r[pos, 1]
          x1, z1 = r[pos, 0] + \
            dr[pos, 0, step], r[pos, 1]+dr[pos, 1, step]
          toCheck = True
          i_BOUNCE = 0
          while toCheck and i_BOUNCE < MAX_BOUNCE:
            toCheck = False
            if i_BOUNCE > 4:
              # print(x1, z1)
              x1 = (x0+x1)/2
              z1 = (z0+z1)/2
            # Left part
            if (x1 < -L/2):
              if ((x1+L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      -L/2, 0., R)
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
              if ((x1-L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      +L/2, 0., R)
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
                # Intersection with x=-L/2
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
            # toCheck=False
          if x1 <= -L/2:
            inside[pos, 0, step] += 1
          elif x1 >= +L/2:
            inside[pos, 1, step] += 1

          if (step + step0) % freq_dumps == 0:
            history[pos, 0, i_DUMP] = x0
            history[pos, 1, i_DUMP] = z0
            i_DUMP += 1

          r[pos, 0], r[pos, 1] = x1, z1

    @cuda.jit(f'void({dtype}[:,:], {dtype}[:,:,:], uint32[:,:,:])')
    def engine(r, dr, inside):
      """Physical engine function compiled in CUDA to simulate pure brownian
      motion particles.

      Args:
        r (float32[:,:]): Initial position of particles 
        dr (float32[:,:,:]): Displacement normaly generated with cupy (way faster than numba for now
                           see https://numba.discourse.group/t/random-array-generation-numba-cuda-slower-than-cupy/815)
        inside (uint32[:,:,:]): ouput array that will store the number of particles in predefinned regions as function of time 
      """
      
      pos = cuda.grid(1)
      if pos < r.shape[0]:
        i_DUMP = 0
        for step in range(dr.shape[2]):
          x0, z0 = r[pos, 0], r[pos, 1]
          x1, z1 = r[pos, 0] + \
            dr[pos, 0, step], r[pos, 1]+dr[pos, 1, step]
          toCheck = True
          i_BOUNCE = 0
          while toCheck and i_BOUNCE < MAX_BOUNCE:
            toCheck = False
            if i_BOUNCE > 4:
              # print(x1, z1)
              x1 = (x0+x1)/2
              z1 = (z0+z1)/2
            # Left part
            if (x1 < -L/2):
              if ((x1+L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      -L/2, 0., R)
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
              if ((x1-L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      +L/2, 0., R)
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
                # Intersection with x=-L/2
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
            # toCheck=False
          if x1 <= -L/2:
            inside[pos, 0, step] += 1
          elif x1 >= +L/2:
            inside[pos, 1, step] += 1

          r[pos, 0], r[pos, 1] = x1, z1
        
    # @cuda.jit(f'void({dtype}[:,:], uint64)')
    @cuda.jit
    def engine_dev(r0, t0, N_steps, sig, inside, rng_states, trajectory, freq_dumps):
      """Physical engine function compiled in CUDA to simulate pure brownian
      motion particles.

      Args:
        r0 (float32[:,2]): Initial position of particles 
        N_steps (uint64): Number of simulation step 
        inside (uint32[:]): ouput array that will store the number of particles in predefinned regions as function of time 
        rng_states (xoroshiro128p_dtype[:]): array of RNG states
        trajectory (float32[:,2,N_dumps]): output trajectory
      """
      N_dumps = trajectory.shape[2] # number of position dump
      # if N_dumps != 0:
      #   freq_dumps = math.ceil(N_steps/N_dumps)

      pos = cuda.grid(1)
      dx, dz = nb.float32(0.0), nb.float32(0.0)
      if pos < r0.shape[0]:  
        x0, z0 = r0[pos, 0], r0[pos, 1]
        i_dump = 0 
        for step in range(N_steps):
          # if step>=473580 and pos == 438:
          #   print('\n')
          #   print(i_dump, step, pos)
          #   print(x0, z0)
          #   print(dx, dz)
          
          dx = sig*xoroshiro128p_normal_float32(rng_states, pos)
          dz = sig*xoroshiro128p_normal_float32(rng_states, pos)
          x1 = x0 + dx
          z1 = z0 + dz
          # if step>=473580 and pos == 438:
          #   print('\n')
          #   print(i_dump, step, pos)
          #   print(x0, z0)
          #   print(x1, z1)
          #   print(-L/2, -h/2)

          toCheck = True
          i_BOUNCE = 0
          while toCheck and i_BOUNCE < MAX_BOUNCE:
            toCheck = False
            if i_BOUNCE > 4:
              # print(x1, z1)
              x1 = (x0+x1)/2
              z1 = (z0+z1)/2
            # Left part
            if (x1 < -L/2):
              if ((x1+L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      -L/2, 0., R)
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
              if ((x1-L/2)**2+z1**2 > R**2):
                x1, z1 = bc.ReflectIntoCircleSimplify(x0, z0,
                                                      x1, z1,
                                                      +L/2, 0., R)
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
                # if step>=473580 and pos == 438:
                #   print('To middle part')
                #   print(x0+L/2, x0-x1)
                #   print((x0+L/2)/(x0-x1))
                #   print('bla')
                # Intersection with x=-L/2
                t = (x0+L/2)/(x0-x1)
                xint = -L/2
                zint = t*z1 + (1-t)*z0
                if math.fabs(zint) > h/2:
                  # if step>=473580 and pos == 438:
                  #   x0,x1 = nb.float32(x0), nb.float32(x1)
                  #   z0,z1 = nb.float32(z0), nb.float32(z1)
                  #   print('bla2')
                  #   NX, NZ = nb.float32(1), nb.float32(0)
                  #   X, Z = nb.float32(-L/2), nb.float32(0)
                  #   t = (NX*(x0-X) + NZ*(z0-Z))/(NX*(x0-x1) + NZ*(z0-z1))
                  #   print(t)
                  #   x_int = x1*t + x0*(1-t)
                  #   z_int = z1*t + z0*(1-t)
                  #   print(x_int, z_int)
                  #   # Finding reflection
                  #   x_1_int, z_1_int = x1-x_int, z1-z_int
                  #   print(x_1_int, z_1_int)
                  #   n_1_int = math.sqrt((x_1_int)**2 + (z_1_int)**2)
                  #   print(n_1_int)
                  #   ux_1_int, uz_1_int = x_1_int/n_1_int, z_1_int/n_1_int
                  #   print(ux_1_int, uz_1_int)
                  #   ps = (ux_1_int*NX + uz_1_int*NZ) # scalar product between n and u_1_int
                  #   print(ps)
                  #   x1p = x_int + n_1_int*(ux_1_int - 2*ps*NX)
                  #   z1p = z_int + n_1_int*(uz_1_int - 2*ps*NZ)
                  #   print(x1p, z1p)
                  #   # print()
                  x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                                x1, z1,
                                                -L/2, 0,
                                                1, 0)
                  # if step>=473580 and pos == 438:
                  #   print(x1, z1)
                  #   print('bla3')

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
            # toCheck=False
          if x1 <= -L/2:
            cuda.atomic.add(inside, step, 1)
          # elif x1 >= +L/2:
          #   cuda.atomic.add(inside, step, 1)
          #   inside[pos, 0, step] += 1
          # elif x1 >= +L/2:
          #   inside[pos, 1, step] += 1
          x0 = x1
          z0 = z1
          if freq_dumps != 0:
            if (step + 1 + t0)%freq_dumps == 0:
              # if (pos==0):
              #   print(step, i_dump)
              trajectory[pos, 0, i_dump] = x0
              trajectory[pos, 1, i_dump] = z0
              i_dump += 1
        r0[pos, 0] = x0
        r0[pos, 1] = z0
      if pos==0:
        t0 += N_steps
        
    self.engine = engine[blockspergrid, threadsperblock]
    self.engine_dev = engine_dev[blockspergrid, threadsperblock]
    self.engine_dump = engine_dump[blockspergrid, threadsperblock]

  def plotPosition(self, s=0.1, **fig_kwargs):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    L, h, R = self.L, self.h, self.R
    # Get current position
    pos = self.pos
    fig, ax = plt.subplots(**fig_kwargs)
    border_kwargs = {'c': 'r'}
    # Draw particles as scatter
    ax.scatter(pos[:, 0], pos[:, 1], s)
    # Draw geometry
    ax.plot(R*np.cos(np.linspace(np.pi/2, 3*np.pi/2, 100))-L/2,
            R*np.sin(np.linspace(np.pi/2, 3*np.pi/2, 100)), **border_kwargs)
    ax.plot(-R*np.cos(np.linspace(np.pi/2, 3*np.pi/2, 100))+L/2,
            R*np.sin(np.linspace(np.pi/2, 3*np.pi/2, 100)), **border_kwargs)
    ax.plot([-L/2, -L/2], [R+h/2, h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [R+h/2, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [h/2, h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [-h/2, -h/2], **border_kwargs)
    ax.plot([-L/2, -L/2], [-R-h/2, -h/2], **border_kwargs)
    ax.plot([+L/2, +L/2], [-R-h/2, -h/2], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')

    return fig, ax

  def run(self, N_steps, freq_dumps=None):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    N_particles = self.N
    batchSize = self._batchSize
    Nbatch = math.floor(N_steps/batchSize)
    D, dt = self.D, self.dt
    step0 = self._step
    dtype = self._dtype
    inside = []
    trajectory = []
    pbar = tqdm(total=N_steps)

    e0 = cp.cuda.Event()
    e1, e2, e3 = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
    dt0 = []
    dt1 = []
    dt2 = []
    pos_GPU = cuda.to_device(self._pos)
    for i in range(Nbatch):
      e0.record()
      _inside = cp.zeros((N_particles, 2, batchSize),
                          dtype=cp.uint32)
      e1.record()
      _dr = cp.random.normal(loc=0, scale=math.sqrt(6*D*dt),
                              size=(N_particles, 2, batchSize),
                              dtype=dtype)
      e2.record()
      if freq_dumps is None:
        self.engine(pos_GPU, _dr, _inside)
      else:
        N_dumps = math.ceil(batchSize/freq_dumps)
        _trajectory = cp.zeros((N_particles, 2, N_dumps),
                                dtype=dtype)
        print(_trajectory.shape)
        self.engine_dump(pos_GPU, _dr, _inside,
                          _trajectory, cp.uint32(self._step))
        print(_trajectory[0, :, 0])
        trajectory.append(_trajectory.get())
      e3.record()
      e3.synchronize()
      inside.append(_inside.sum(axis=0).get())
      self._step += batchSize
      pbar.update(batchSize)
      pbar.set_postfix_str(f'{prefix(i*batchSize*self.dt*1E-15)}s')
      dt0.append(cp.cuda.get_elapsed_time(e0, e1))
      dt1.append(cp.cuda.get_elapsed_time(e1, e2))
      dt2.append(cp.cuda.get_elapsed_time(e2, e3))
    inside = np.hstack(inside)
    dt0, dt1, dt2 = np.mean(dt0[2:]), np.mean(dt1[2:]), np.mean(dt2[2:])
    dt_total = dt0+dt1+dt2
    print('\n')
    print(f'With {N_particles} particles')
    print(f'GPU time per step:')
    print(f'cp.zeros: {dt0/batchSize*1E3:.3f} us')
    print(f'cp.random.normal: {dt1/batchSize*1E3:.3f} us')
    print(f'engine: {dt2/batchSize*1E3:.3f} us')
    print(f'Total: {dt_total/batchSize*1E3:.3f} us')
    print(f'Need {(dt_total/batchSize)*(1E12/dt)*1E-3:.2f}s to compute 1ms of simulation')
    rootgrp = Dataset(self.output_path,
                      "a", format="NETCDF4")

    if 'run' not in rootgrp.groups:
      rootgrp.createGroup(f'run')
    runsgrp = rootgrp['run']

    i = 0
    while f'{i}' in runsgrp.groups:
        i += 1
    rungrp = runsgrp.createGroup(f'{i}')
    rungrp.status = 'SUCCEED'
    rungrp.step_i = step0
    rungrp.step_f = self._step
    rungrp.createDimension("step", N_steps)
    for i in range(_inside.shape[1]):
      # regiongrp = rungrp.createGroup(f'region_{i}')
      # inside_region = regiongrp.createVariable("N","u4",("idx","step"))
      inside_region = rungrp.createVariable(
          f'region_{i}', "u4", ("step"))
      inside_region[:] = inside[i, :]
    rungrp['region_0'].definition = 'x<=L/2'
    rungrp['region_1'].definition = 'x>=L/2'

    if freq_dumps is not None:
      rootgrp.createDimension("step", None)
      trajectory = np.concatenate(trajectory, axis=2)

      rungrp.createDimension("dump_step", N_steps//freq_dumps+1)
      trajvar = rungrp.createVariable(
          f'trajectory', "f4", ("idx", "dim", "dump_step"), zlib=True)
      trajvar[:, :, :] = trajectory[:, :, :]
      print(trajectory[0, :, 0])
      print(trajvar[0, :, 0])
      trajvar.freq_dumps = freq_dumps
    rootgrp.close()

    del pos_GPU, _inside
    self._mempool.free_all_blocks()
    self._pinned_mempool.free_all_blocks()
    return True

  def run2(self, N_steps, freq_dumps=0):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    chunk_size = 100_000 #dask ???

    # TODO : add a warmup
    
    freq_dumps = nb.uint32(freq_dumps)
    N_particles = self.N
    D, dt = self.D, self.dt
    dtype = self._dtype
    
    N_steps = nb.uint64(N_steps)
    step0 = self._step
    # pbar = tqdm(total=N_steps)
    scale = nb.float32(math.sqrt(2*D*dt))
    if freq_dumps==0:
      trajectory = np.zeros((N_particles,2,0), dtype=np.float32)
    else:
      N_dumps = math.floor(N_steps/freq_dumps)+1
      trajectory = np.zeros((N_particles,2,N_dumps), dtype=np.float32)
    
    inside = np.zeros(N_steps, np.uint32)
    
    e0 = cp.cuda.Event()
    e1, e2, e3 = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
    dt0 = []
    dt1 = []
    dt2 = []
    e0.record()
    d_trajectory =  cuda.to_device(trajectory)
    dpos = cuda.to_device(self._pos)
    d_inside = cuda.to_device(inside)
    rng_states = create_xoroshiro128p_states(N_particles, seed=1)
    e1.record()
    self.engine_dev(dpos, N_steps, scale, d_inside, rng_states, d_trajectory, freq_dumps)
    e2.record()
    d_inside.copy_to_host(inside)
    d_trajectory.copy_to_host(trajectory)
    e3.record()
    e3.synchronize()
    dt0 = cp.cuda.get_elapsed_time(e0, e1)
    dt1 = cp.cuda.get_elapsed_time(e1, e2)
    dt2 = cp.cuda.get_elapsed_time(e2, e3)
    dt_total = dt0+dt1+dt2
    print('\n')
    print(f'With {N_particles} particles')
    print(f'GPU time per step:')
    print(f'cp.zeros: {dt0/N_steps*1E3:.3f} us')
    print(f'cp.random.normal: {dt1/N_steps*1E3:.3f} us')
    print(f'engine: {dt2/N_steps*1E3:.3f} us')
    print(f'Total: {dt_total/N_steps*1E3:.3f} us')
    print(f'Need {(dt_total/N_steps)*(1E12/dt)*1E-3:.2f}s to compute 1ms of simulation')

    rootgrp = Dataset(self.output_path,
                      "a", format="NETCDF4")

    
    if 'run' not in rootgrp.groups:
      rootgrp.createGroup(f'run')
    runsgrp = rootgrp['run']

    i = 0
    while f'{i}' in runsgrp.groups:
        i += 1
    rungrp = runsgrp.createGroup(f'{i}')
    rungrp.status = 'SUCCEED'
    rungrp.step_i = step0
    rungrp.step_f = self._step
    rungrp.createDimension("step", N_steps)
    inside_region = rungrp.createVariable(
          'region_0', "u4", ("step"))
    inside_region.definition = 'x<=L/2'
    inside_region[:] = inside[:]

    if freq_dumps!=0:
      rungrp.createDimension("dump_step", N_dumps)
      trajvar = rungrp.createVariable(
          f'trajectory', "f4", ("idx", "dim", "dump_step"), zlib=True)
      # print(trajvar.shape, trajectory.shape)
      trajvar[:, :, :] = trajectory[:, :, :]
      trajvar.freq_dumps = freq_dumps

    rootgrp.close()
    del d_trajectory, d_inside
    # self._mempool.free_all_blocks()
    # self._pinned_mempool.free_all_blocks()
    return True
  
  def run3(self, N_steps, freq_dumps=0):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    with Dataset(self.output_path,
                      "a", format="NETCDF4") as rootgrp:

      if 'run' not in rootgrp.groups:
        rootgrp.createGroup(f'run')
      runsgrp = rootgrp['run']

      i = 0
      while f'{i}' in runsgrp.groups:
          i += 1
      rungrp = runsgrp.createGroup(f'{i}')
      rungrp.status = 'UNCOMPLET'
      rungrp.step_i = self._step
      rungrp.step_f = self._step + N_steps
      rungrp.createDimension("step", N_steps)
      inside_region = rungrp.createVariable(
            'region_0', "u4", ("step"))
      inside_region.definition = 'x<=L/2'

      if freq_dumps!=0:
        N_dumps = math.floor(N_steps/freq_dumps)
        rungrp.createDimension("dump_step", N_dumps)
        trajvar = rungrp.createVariable(
            f'trajectory', "f4", ("idx", "dim", "dump_step"), zlib=True)
        trajvar.freq_dumps = freq_dumps

    max_chunk_size = 100_000 # TODO : dask ???

    # TODO : add a warmup
    N_steps = nb.uint64(N_steps)
    freq_dumps = nb.uint32(freq_dumps)
    N_particles = self.N
    D, dt = self.D, self.dt
    scale = nb.float32(math.sqrt(2*D*dt))
    dtype = self._dtype

    # Transfert current position to device
    dpos = cuda.to_device(self._pos)

    rng_states = create_xoroshiro128p_states(N_particles, seed=self._initial_seed) # TODO : need to change that after each run !

    inside = np.zeros(N_steps, np.uint32)
    if freq_dumps==0:
      trajectory = np.zeros((N_particles,2,0), dtype=np.float32)
    else:
      N_dumps = math.floor(N_steps/freq_dumps)
      trajectory = np.zeros((N_particles, 2, N_dumps), dtype=np.float32)

    # i_step = self._step 
    pbar = tqdm(total=math.ceil(N_steps))
    for i_chunk, i_step in enumerate(range(0, N_steps, max_chunk_size)):
      chunck_interval = range(0, N_steps)[i_step:i_step + max_chunk_size]
      chunk_size = chunck_interval[-1] - chunck_interval[0] + 1

      # Allocate device memory to store trajectory
      if freq_dumps==0:
        i_trajectory = np.zeros((N_particles,2,0), dtype=np.float32)
      else:
        N_dumps = math.floor(chunk_size/freq_dumps)
        i_trajectory = np.zeros((N_particles, 2, N_dumps), dtype=np.float32)
      d_trajectory =  cuda.to_device(i_trajectory)

      # Allocate device memory to store number of particle in region 0
      i_inside = np.zeros(chunk_size, np.uint32)
      d_inside = cuda.to_device(i_inside)

      self.engine_dev(dpos, self._step, chunk_size, scale, d_inside, 
                      rng_states, d_trajectory, freq_dumps)

      d_inside.copy_to_host(inside[i_step:i_step + max_chunk_size])
      if freq_dumps!=0:
        d_trajectory.copy_to_host(i_trajectory)
        N_dumps = math.floor(max_chunk_size/freq_dumps)
        trajectory[:,:,N_dumps*i_chunk:N_dumps*(i_chunk + 1)] = i_trajectory

      self._step += chunk_size
      pbar.set_postfix(total = f'{prefix(self._step*self.dt*1E-15)}s')
      pbar.update(chunk_size)
      
    pass
    with Dataset(self.output_path, "a", 
                  format="NETCDF4") as rootgrp:
      rootgrp[f'run/{i}/region_0'][:] = inside
      if freq_dumps!=0:
        rootgrp[f'run/{i}/trajectory'][:] = trajectory
    # e0 = cp.cuda.Event()
    # e1, e2, e3 = cp.cuda.Event(), cp.cuda.Event(), cp.cuda.Event()
    # dt0 = []
    # dt1 = []
    # dt2 = []
    # e0.record()
    # d_trajectory =  cuda.to_device(trajectory)
    # dpos = cuda.to_device(self._pos)
    # 
    # rng_states = create_xoroshiro128p_states(N_particles, seed=1)
    # e1.record()
    # 
    # e2.record()
    # d_inside.copy_to_host(inside)
    # d_trajectory.copy_to_host(trajectory)
    # e3.record()
    # e3.synchronize()
    # dt0 = cp.cuda.get_elapsed_time(e0, e1)
    # dt1 = cp.cuda.get_elapsed_time(e1, e2)
    # dt2 = cp.cuda.get_elapsed_time(e2, e3)
    # dt_total = dt0+dt1+dt2
    # print('\n')
    # print(f'With {N_particles} particles')
    # print(f'GPU time per step:')
    # print(f'cp.zeros: {dt0/N_steps*1E3:.3f} us')
    # print(f'cp.random.normal: {dt1/N_steps*1E3:.3f} us')
    # print(f'engine: {dt2/N_steps*1E3:.3f} us')
    # print(f'Total: {dt_total/N_steps*1E3:.3f} us')
    # print(f'Need {(dt_total/N_steps)*(1E12/dt)*1E-3:.2f}s to compute 1ms of simulation')

    # rootgrp = Dataset(self.output_path,
    #                   "a", format="NETCDF4")

    
    # if 'run' not in rootgrp.groups:
    #   rootgrp.createGroup(f'run')
    # runsgrp = rootgrp['run']

    # i = 0
    # while f'{i}' in runsgrp.groups:
    #     i += 1
    # rungrp = runsgrp.createGroup(f'{i}')
    # rungrp.status = 'SUCCEED'
    # rungrp.step_i = step0
    # rungrp.step_f = self._step
    # rungrp.createDimension("step", N_steps)
    # inside_region = rungrp.createVariable(
    #       'region_0', "u4", ("step"))
    # inside_region.definition = 'x<=L/2'
    # inside_region[:] = inside[:]

    # if freq_dumps!=0:
    #   rungrp.createDimension("dump_step", N_dumps)
    #   trajvar = rungrp.createVariable(
    #       f'trajectory', "f4", ("idx", "dim", "dump_step"), zlib=True)
    #   # print(trajvar.shape, trajectory.shape)
    #   trajvar[:, :, :] = trajectory[:, :, :]
    #   trajvar.freq_dumps = freq_dumps

    # rootgrp.close()
    # del d_trajectory, d_inside
    # # self._mempool.free_all_blocks()
    # # self._pinned_mempool.free_all_blocks()
    # return True


if __name__ == "__main__":
  dt = int(1E6) #fs (1ns) - time steps
  D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

  # Geometry
  L = 1E3 # A (100nm) - channel length
  h = 1E2 # A (10nm)  - channel height
  R = 1E4 # A (1um) - reservoir size

  N= 8*1024

  u = Universe(N=N, L=L, h=h, R=R, D=D, dt=dt,
              output_path='simu.nc')

  u.run3(48_023, freq_dumps=10);
