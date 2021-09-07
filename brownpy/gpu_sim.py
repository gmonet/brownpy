import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from netCDF4 import Dataset
import h5py
from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_normal_float32)
from numpy.random.mtrand import seed
from tqdm.auto import tqdm
import cupy as cp

from brownpy import bc
from brownpy import topology
from brownpy.utils import prefix
from brownpy.settings import _GPU_COMPUTATION

# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# def set_

class Universe():
  """The Universe contains all the information describing the system.

  It contains information about the topology of the system as the
  geometry, the timestep, the number of particles, their diffusion coefficient,
  the boundary condition ...

  Properties created with the ``@property`` decorator should be documented
  in the property's getter method.

  Attributes:
    attr1 (str): Description of `attr1`.
    attr2 (:obj:`int`, optional): Description of `attr2`.

  """
  __version__ = '0.0.9a'
  MAX_BOUNCE = 10

  def __init__(self,
               top: topology.Topology,
               N: int,
               D: float, dt: int,
               output_path: str,
               overwrite=False,
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
      overwrite (bool): if false iterate the name of output file to avoid overwriting
    kwargs (advanced parameters):
      gpu_memory (int, optional): GPU memory size to use. Default to 2*1024**3 (2GiB)
      threadsperblock (int, optional): Number of thread per block. Default to 128
      seed (int): set seed
    """
    # Define timestep
    self._dt = dt
    self._step = 0  # current step

    # Define topology
    self._top = top

    # Define number of particles and their properties
    self._N = N
    self._D = D
    
    # Fix computation precision (TODO: make it compatible with saving in netcdf file)
    self._dtype = 'float32'

    # Cut the time axis into batch of gpu-computation in order to fill the gpu memory
    # 2GiB in default #TODO : not properly handled
    gpu_memory = kwargs.get('gpu_memory', 2*1024**3)
    self._batchSize = int(gpu_memory/4/2/N)  # Batch size in timestep

    # Thread per block (128 by default)
    self._threadsperblock = kwargs.get('threadsperblock', 128)
    if self.N % self._threadsperblock != 0:
      RuntimeWarning(
        f"""The number of particles should be a multiple of {self._threadsperblock} 
        for optimal performance""") #TODO : check if everyting work in this case
    self._blockspergrid = math.ceil(self._N / self._threadsperblock)

    # Set seed if defined by user or use the system clock as seed
    self._initial_seed = kwargs.get('seed', time.time_ns())
    np.random.seed(self._initial_seed%(2**32-1))

    # Fill randomly the geometry
    self._pos = top.fill_geometry(N, seed=self._initial_seed)
    
    # Compile physical engine
    self._compile()
    
    self._output_path = output_path
    if not kwargs.get('_outputFileProvided', False):
      # Create a netcdf simulation file
      self._initOutputFile(output_path, overwrite)
  
  @classmethod
  def from_hdf5(cls, input_path: str):
    """Create Universe from hdf5 file previously created

    Args:
      input_path (str): hdf5 file
    """

    input_path = Path(input_path).resolve()
    if not input_path.exists():
      raise FileNotFoundError(f"{str(input_path)} doesn't exist.")

    with h5py.File(str(input_path), "r") as f:
      top_name = f['geometry'].attrs['name']
      top_class = getattr(topology, top_name)
      if f.attrs['version'] != cls.__version__:
        raise DeprecationWarning('Depreciated version of h5py file')

      top = top_class.from_hdf5(f['geometry'])
      u = cls(top=top,
              N=f['particles'].attrs['N'],
              D=f['particles']['0'].attrs['D'],
              dt=f.attrs['dt'],
              output_path=input_path,
              _outputFileProvided=True)
    return u

  #region Properties
  @property
  def pos(self):
    """array: current position of particles (in A)."""
    return self._pos

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
    with h5py.File(self._output_path, "r") as f:
      N_runs = f['run'].attrs['N_runs']
    return N_runs

  def __getitem__(self, key):
    data = {}
    with h5py.File(self._output_path, "r") as f:
      groups_keys = list(f['run'])

    if isinstance(key, int):
      key = str(key)

    if isinstance(key, str):
      split = 1000
      if key not in groups_keys:
        raise KeyError(f"Available runs : {', '.join(groups_keys)}")
      with h5py.File(self._output_path, "r") as f:
        if f'run/{key}/trajectory' in f:
          print(f'Reading trajectory ...')
          value = f[f'run/{key}/trajectory']
          data['trajectory'] = value[...]
          print(f'... Done')
        for name, value in f[f'run/{key}/regions/'].items():
          data[name] = value[...]
          print(f'{name}  read')
          # TODO: Test chunck with h5py
          # data[key] = np.empty(shape = value.shape, 
          #                      dtype = value.dtype)
          # if value.ndim==1: # region variable
          #   total = value.shape[0]
          #   pbar = tqdm(total=total)
          #   for i in range(0, total, split):
          #     value[i:i+split]
      return data
    else:
      raise TypeError(f'universe indices must be integers or str, not {type(key)}')
  
  def _initOutputFile(self, output_path: str, overwrite:bool):
    """Create and initialize the netcdf simulation file
    Args:
      output_path (str): Output netcdf file
      overwrite (bool): if false iterate the name of output file to avoid overwriting
    """
    # Check if the file already exists.
    # If yes, the file name will be incremented.
    output_path = Path(output_path).resolve()
    output_path = output_path.with_suffix('.hdf5')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and overwrite is False:
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

    # # Create hdf5 file
    with h5py.File(self.output_path, 'w') as f:
      f.attrs['source'] = 'Created with BM_cuda'
      f.attrs['version'] = self.__version__
      f.attrs['date'] = "Created " + time.ctime(time.time())
      f.attrs['units'] = 'real'  # We use LAMMPS real units
      f.attrs['dt'] = self.dt
      f.attrs['dtype'] = self._dtype
      f.attrs['ndim'] = 2 # Specify that it is a 2D simulation

      particlesgrp = f.create_group('particles')
      particlesgrp.attrs['N'] = self.N # Specify total number of particles
      particlesgrp.attrs['_seed'] = self._initial_seed
      particlesgrp.create_dataset("initial_pos", data=self.pos)
      particlesgrp.create_dataset("type", data=[0]*self.N, dtype=np.uint8)
      particlegrp_0 = particlesgrp.create_group('0')
      particlegrp_0.attrs['type'] = 0
      particlegrp_0.attrs['D'] = self.D

      # Create a group holding informations about geometry
      geomgrp = f.create_group("geometry")
      self._top.to_hdf5(geomgrp)

      # Create a group that will hold informations about simulations
      rungrp = f.create_group("run")
      rungrp.attrs['N_runs'] = 0  # number of simulation perfomed until now

  def _compile(self):
    """Compile kernel cuda function
    """
    compute_boundary_condition = self._top.compute_boundary_condition
    check_region = self._top.check_region
    scale = nb.float32(math.sqrt(2*self.D*self.dt))
    if _GPU_COMPUTATION:
      @cuda.jit
      def engine(r0, t0, N_steps, inside, rng_states, trajectory, freq_dumps,
                _internal_states):
        """Physical engine function compiled in CUDA to simulate pure brownian
        motion particles.

        Args:
          r0 (float32[:,2]): Initial position of particles 
          N_steps (uint64): Number of simulation step 
          inside (uint32[:,:]): ouput array that will store the number of particles in predefinned regions as function of time 
          rng_states (xoroshiro128p_dtype[:]): array of RNG states
          trajectory (float32[:,2,N_dumps]): output trajectory
        """
        N_dumps = trajectory.shape[2] # number of position dump
        # freq_dumps = 0
        # if N_dumps != 0:
        #   freq_dumps = math.ceil(N_steps/N_dumps)

        pos = cuda.grid(1)
        dx, dz = nb.float32(0.0), nb.float32(0.0)
        if pos < r0.shape[0]:
          x0, z0 = r0[pos, 0], r0[pos, 1]
          i_dump = 0 

          internal_state = cuda.local.array(1, nb.uint32)
          for i in range(_internal_states.shape[1]):
            internal_state[i] = _internal_states[pos, i]

          for step in range(N_steps):
            dx = scale*xoroshiro128p_normal_float32(rng_states, pos)
            dz = scale*xoroshiro128p_normal_float32(rng_states, pos)
            x1 = x0 + dx
            z1 = z0 + dz

            x1, z1 = compute_boundary_condition(x0, z0, x1, z1, rng_states, internal_state)
            check_region(x1, z1, inside, step, internal_state)

            x0 = x1
            z0 = z1
            if freq_dumps != 0:
              if (step + 1 + t0)%freq_dumps == 0:
                trajectory[pos, 0, i_dump] = x0
                trajectory[pos, 1, i_dump] = z0
                i_dump += 1
          r0[pos, 0] = x0
          r0[pos, 1] = z0

          for i in range(_internal_states.shape[1]):
            _internal_states[pos, i] = internal_state[i]

        if pos==0:
          t0 += N_steps
      
      self.engine = engine
    else:
      @nb.njit(parallel=True)
      def engine(r0, t0, N_steps, inside, rng_states, trajectory, freq_dumps,
                _internal_states):
        """Physical engine function compiled in CUDA to simulate pure brownian
        motion particles.

        Args:
          r0 (float32[:,2]): Initial position of particles 
          N_steps (uint64): Number of simulation step 
          inside (uint32[:,:]): ouput array that will store the number of particles in predefinned regions as function of time 
          rng_states (xoroshiro128p_dtype[:]): array of RNG states
          trajectory (float32[:,2,N_dumps]): output trajectory
        """
        N_dumps = trajectory.shape[2] # number of position dump
        dx, dz = nb.float32(0.0), nb.float32(0.0)
        for pos in nb.prange(r0.shape[0]):
          x0, z0 = r0[pos, 0], r0[pos, 1]
          i_dump = 0 

          internal_state = np.zeros(1, dtype=nb.uint32)
          for i in range(_internal_states.shape[1]):
            internal_state[i] = _internal_states[pos, i]

          for step in range(N_steps):
            dx = scale*np.random.standard_normal()
            dz = scale*np.random.standard_normal()
            x1 = x0 + dx
            z1 = z0 + dz

            x1, z1 = compute_boundary_condition(x0, z0, x1, z1, rng_states, internal_state)
            check_region(x1, z1, inside, step, internal_state)

            x0 = x1
            z0 = z1
            if freq_dumps != 0:
              if (step + 1 + t0)%freq_dumps == 0:
                trajectory[pos, 0, i_dump] = x0
                trajectory[pos, 1, i_dump] = z0
                i_dump += 1
          r0[pos, 0] = x0
          r0[pos, 1] = z0

          for i in range(_internal_states.shape[1]):
            _internal_states[pos, i] = internal_state[i]

        t0 += N_steps
      
      self.engine = engine
  
  def plot(self, ax=None, scatter_kwargs={'s': 0.1}):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    # Get current position
    pos = self.pos
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    # Draw particles as scatter
    ax.scatter(pos[:, 0], pos[:, 1], **scatter_kwargs)
    
    # Draw geometry
    top = self._top
    top.plot(ax)

    return fig, ax

  def run(self, N_steps, freq_dumps=0, **kwargs):
    """Plot current position of particles

    Args:
      s (float, optional): size of scatter. Default to 0.1
      fig_kwargs (optional): Paramter to pass to plt.subplots
    """
    

    max_chunk_size = min(N_steps, 100_000) # TODO : dask ???

    with h5py.File(self._output_path, 'a') as f:
      runsgrp = f['run']

      i_run = 0
      while f'{i_run}' in runsgrp:
          i_run += 1
      rungrp = runsgrp.create_group(f'{i_run}')
      rungrp.attrs['status'] = 'UNCOMPLETED'
      rungrp.attrs['step_i'] = self._step
      rungrp.attrs['step_f'] = self._step + N_steps
      
      N_regions = len(self._top.regions)
      regionsgrp = rungrp.create_group("regions")
      regions_ds = []
      for region in self._top.regions:

        region_ds = regionsgrp.create_dataset(region['name'], dtype=np.uint16, 
                                              shape=(N_steps,),
                                              chunks=(max_chunk_size,))
        region_ds.attrs['definition'] = region['def']
        regions_ds.append(region_ds)

      freq_dumps = nb.uint32(freq_dumps)
      if freq_dumps!=0:
        N_dumps = math.floor(N_steps/freq_dumps)
        max_dumps_per_chunk = math.floor(max_chunk_size/freq_dumps)
        traj_ds = rungrp.create_dataset('trajectory', dtype=self._dtype, 
                                        shape=(self.N,2,N_dumps),
                                        chunks=(self.N,2,max_dumps_per_chunk)
                                        # chunks=(self.N,2,N_dumps)
                                        # ValueError: Number of elements in chunk must be < 4gb (number of elements in chunk must be < 4GB)
                                        )
        rungrp
        # TODO: resize faster
        traj_ds.attrs['freq_dumps'] = freq_dumps
      else:
        N_dumps = 0
    
    N_steps = nb.uint64(N_steps) # Total number of steps

    N_chunks = math.ceil(N_steps/max_chunk_size) # Number of chunk
      
    N_particles = self.N
    D, dt = self.D, self.dt
    dtype = self._dtype

    # Get current position of particles
    pos = self._pos

    

    # Allocate memory to store internal state for each particles
    internal_states = np.zeros((N_particles, 1), dtype=np.uint32)

    # Allocate memory to store trajectory
    N_dumps = 0 if freq_dumps==0 else math.floor(max_chunk_size/freq_dumps)
    trajectory = np.zeros((N_particles, 2, N_dumps), dtype=np.float32)

    if _GPU_COMPUTATION:
      # Transfert current position to device
      d_pos = cuda.to_device(pos)
      # TODO : use pinned array ?
      
      # Allocate device memory to store number of particle in regions
      
      d_internal_states = cuda.to_device(internal_states) # Transfert to device memory

      # Allocate device memory to store trajectory
      d_trajectory = cuda.to_device(trajectory) # Transfert to device memory

      # Create individual random generator states for each CUDA thread 
      # Note: max independant number generation : 2**64 (1.8E19)
      self._initial_seed = np.uint64(kwargs.get('seed', time.time_ns())%(2**64-1))
      rng_states = create_xoroshiro128p_states(N_particles, seed=self._initial_seed)

      # i_step = self._step 
      e0, e1, e2, e3 = cuda.event(), cuda.event(), cuda.event(), cuda.event()
      # transfertStream1 = cuda.stream()
      # transfertStream2 = cuda.stream()
      dt1, dt2, dt3 = [], [], []
    else:
      rng_states=np.zeros(N_particles, dtype=np.uint32)

    t0_cpu = time.perf_counter()
    pbar = tqdm(total=math.ceil(N_steps))
    for i_chunk, i_step in enumerate(range(0, N_steps, max_chunk_size)):
      chunck_interval = range(0, N_steps)[i_step:i_step + max_chunk_size]
      chunk_size = chunck_interval[-1] - chunck_interval[0] + 1

      if _GPU_COMPUTATION:
        e0.record()
        # Allocate device memory to store number of particle in regions
        d_p_inside=cp.zeros((N_regions, max_chunk_size), dtype=np.uint32)
        e1.record()
        self.engine[self._blockspergrid, 
                    self._threadsperblock](d_pos, # r0
                                            self._step, # t0 
                                            chunk_size, # N_steps
                                            d_p_inside, # inside 
                                            rng_states, # rng_states
                                            d_trajectory, # trajectory
                                            freq_dumps, #freq_dumps
                                            d_internal_states,
                                            )
        e2.record()
        # Transfert results from device to RAM
        # TODO : USE STREAM !!!
        if freq_dumps!=0: d_trajectory.copy_to_host(trajectory)
        p_inside = d_p_inside.get()
        e3.record()
        cuda.synchronize()
        dt1.append(cuda.event_elapsed_time(e0,e1)*1E-3)
        dt2.append(cuda.event_elapsed_time(e1,e2)*1E-3)
        dt3.append(cuda.event_elapsed_time(e2,e3)*1E-3)
      else:
        p_inside = np.zeros((N_regions, max_chunk_size), dtype=np.uint32)
        self.engine(pos, # r0
                    self._step, # t0 
                    chunk_size, # N_steps
                    p_inside, # inside 
                    rng_states, # rng_states
                    trajectory, # trajectory
                    freq_dumps, #freq_dumps
                    internal_states,
                    )
      # Transfert result from RAM to drive
      with h5py.File(self._output_path, 'a') as f:
        for i, region in enumerate(self._top.regions):
          region_ds = f[f'run/{i_run}/regions/{region["name"]}']
          regions_ds[i_step:i_step + max_chunk_size] = p_inside[i, :chunk_size] 
        if freq_dumps!=0:
          traj_ds = f[f'run/{i_run}/trajectory']
          i_N_dumps = math.floor(chunk_size/freq_dumps)
          traj_ds[:,:,max_dumps_per_chunk*i_chunk:max_dumps_per_chunk*(i_chunk + 1)] = trajectory[:,:,:i_N_dumps]

      self._step += chunk_size
      pbar.set_postfix(total = f'{prefix(self._step*self.dt*1E-15)}s')
      pbar.update(chunk_size)
    pbar.close()
    if _GPU_COMPUTATION: del d_trajectory, d_pos; cuda.synchronize()
    del trajectory
    t1_cpu = time.perf_counter()
    dt_cpu = t1_cpu - t0_cpu
    
    print(f'With {N_particles} particles')
    if _GPU_COMPUTATION: 
      dt1, dt2, dt3 = N_chunks*np.mean(dt1), N_chunks*np.mean(dt2), N_chunks*np.mean(dt3)
      print(f'------------------------------------------')
      print(f'GPU time per step and per particles:')
      print(f'Allocation: {prefix(dt1/N_steps/N_particles)}s')
      print(f'Engine: {prefix(dt2/N_steps/N_particles)}s')
      print(f'Transfert to RAM: {prefix(dt3/N_steps/N_particles)}s')
      print(f'Total: {prefix((dt1+dt2+dt3)/N_steps/N_particles)}s')
    print(f'------------------------------------------')
    print(f'CPU time per step and per particles:')
    print(f'Total: {prefix(dt_cpu/N_steps/N_particles)}s')
    print(f'------------------------------------------')
    print(f'For a timestep of {prefix(dt*1E-15)}s')
    print(f'To simulate the trajectory of 1 particle during 1 s, we need {prefix((dt_cpu/N_steps/N_particles)*(1E15/dt))}s')
    
    # return True


if __name__ == "__main__":

  dt = int(1E6) #fs (1ns) - time steps
  D = 1.5E-4 # A²/fs  (1.5E-9 m²/s) - Diffusion coefficient

  # Geometry
  L = 1E3 # A (100nm) - channel length
  h = 1E2 # A (10nm)  - channel height
  R = 1E4 # A (1um) - reservoir size

  N= 4*1024

  top = topology.ElasticChannel1(L=L, h=h, R=R)
  u = Universe(N=N, top=top, D=D, dt=dt,
              output_path='simu', overwrite=True)
  u.run(10000, freq_dumps=100)
  # u2 = Universe.from_hdf5(u.output_path)
