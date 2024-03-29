
import math
import textwrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, float32, ndarray, uint, uint32
import numba as nb
from numba import cuda, jit
from numba.cuda.random import xoroshiro128p_uniform_float32

from brownpy.geometry import vector2D
from brownpy import bc

dtype = float32

# Maximum bounce during one step
# It may happen that the elastic bounce of a particle lead it to another wall
MAX_BOUNCE = 4
get_random_uniform = None

class Topology():
  def __init__(self) -> None:
    self._previous_gen_settings = {}

  def __str__(self) -> str:
    text = self.__class__.__name__ + '\n'
    text += f'version {self.__class__.__version__}\n'
    for key, value in self.__dict__.items():
      if isinstance(value, (bool, int, float, str)):
        text += f'{key}={value}\n'
    return text

  def gen_jitted_functions(self, gen_settings):
    '''
    Genreate jitted function as a function of some settings

    settings:
      regions (list or None): if None used default top regions
      target (str): gpu or cpu
    Create device CUDA function from defined regions
    Args:
      regions (list of dict): Each input must be dictionnary with, at least,
        a 'name' and 'def' keys.
    Returns:
      check_region (cuda device function)
    '''
    # If settings for generating function didn't change, we keep previously defined
    # function
    if gen_settings['regions'] == self._previous_gen_settings.get('regions') and \
            gen_settings['target'] == self._previous_gen_settings.get('target'):
      return

    target = gen_settings['target']
    regions = gen_settings['regions']
    if regions is None:
      regions = self.regions

    if target == 'gpu':
      if not cuda.is_available():
        raise SystemError('CUDA is not availabled on this system')
      code_check_region = f'''
      @cuda.jit(device=True)
      def check_region(x:nb.types.float32, z:nb.types.float32,
                        inside:nb.types.Array, 
                        step:nb.types.uint64, 
                        internal_state:tuple) -> None:
        pass
      '''
      for i, region in enumerate(regions):
        code_check_region += f'''
        if {region['def']}:
          cuda.atomic.add(inside, ({i}, step), 1)
        '''

      @jit(nopython=True, device=True)
      def _get_random_uniform(rng_states):
        pos = cuda.grid(1)
        return xoroshiro128p_uniform_float32(rng_states, pos)

    elif target == 'cpu':
      code_check_region = f'''
      @jit(nopython=True)
      def check_region(x:nb.types.float32, z:nb.types.float32,
                        inside:nb.types.Array, 
                        step:nb.types.uint64, 
                        internal_state:tuple) -> None:
        pass
      '''
      for i, region in enumerate(regions):
        code_check_region += f'''
        if {region['def']}:
          inside[{i}, step] += 1
        '''

      @jit(nopython=True)
      def _get_random_uniform(rng_states):
        return np.random.standard_normal()
    else:
      raise ValueError('Target argument should be cpu or gpu')

    code_check_region = textwrap.dedent(code_check_region)
    input_dict = globals()
    for key, value in self.__dict__.items():
      if isinstance(value, (bool, int, float, str)):
        input_dict[key] = value
    return_dict = {}
    exec(code_check_region, input_dict, return_dict)

    self.check_region = return_dict['check_region']
    
    global get_random_uniform
    get_random_uniform = _get_random_uniform

    self._previous_gen_settings = gen_settings

  def fill_geometry(self, N: uint):
    """Randomly fill the geometry
    """
    raise NotImplementedError

  def to_hdf5(self, geom_grp: h5py.Group):
    geom_grp.attrs['name'] = self.__class__.__name__
    geom_grp.attrs['version'] = self.__class__.__version__
    for key, value in self.__dict__.items():
      if isinstance(value, (bool, int, float, str)):
        geom_grp.attrs[key] = value

  @classmethod
  def from_hdf5(cls, geom_grp: h5py.Group):
    dic = geom_grp.attrs
    return cls(**dic)

  def compute_boundary_condition(self,
                                 x0: dtype, z0: dtype,
                                 x1: dtype, z1: dtype,
                                 rng_states: array,
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
    raise NotImplementedError

  def check_region(self,
                   x: dtype, z: dtype,
                   inside: nb.types.Array,
                   step: nb.types.uint64,
                   internal_state: tuple) -> None:
    raise NotImplementedError

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    raise NotImplementedError

  def plot(self, ax=None):
    raise NotImplementedError


class Infinite(Topology):
  __version__ = '0.0.2'

  def __init__(self, **kwargs) -> None:
    """Just an infinite space without any walls
    Args:
      None
    """
    regions = [{'name': 'left', 'def': 'x<=0'}]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

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


class InfiniteSlitAbsorbing(Topology):
  __version__ = '0.0.2'

  def __init__(self, L: dtype, h: dtype, l: float, **kwargs) -> None:
    """Inifinite slit with absorbing walls

    ━━━━━━━━━━━━━━━━━━━━━ ↑  
                          │
                          │
                          │ h
                          │
                          │
    ━━━━━━━━━━━━━━━━━━━━━ ↓ 
    ←-------------------→
              L   
    ━━ : Absorbing wall

    Args:
      L (float in A): Length of the channel
      h (float in A): Height of the channel
      l (float in dt-1) : Desorption frequency  
    """
    self.L, self.h = L, h
    self.l = l
    regions = [{'name': 'absorbed', 'def': 'internal_state[0] > 0'}]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):

      if internal_state[0] != 0:
        internal_state[0] -= 1
        x1 = x0
        z1 = z0
        return x1, z1

      # Intersection with bottom channel
      X, Z = 0, -h/2
      NX, NZ = 0, 1
      if z1 < Z:
        den = (x1-x0)*NX + (z1-z0)*NZ
        if den != 0:
          t = ((X-x0)*NX + (Z-z0)*NZ)/den
          xint = t*x1 + (1-t)*x0
          zint = t*z1 + (1-t)*z0
          x1, z1 = xint, zint
          T = -(1/l)*math.log(1-get_random_uniform(rng_states))
          internal_state[0] = uint32(T)

      # Intersection with top channel
      X, Z = 0, +h/2
      NX, NZ = 0, 1
      if z1 > Z:
        den = (x1-x0)*NX + (z1-z0)*NZ
        if den != 0:
          t = ((X-x0)*NX + (Z-z0)*NZ)/den
          xint = t*x1 + (1-t)*x0
          zint = t*z1 + (1-t)*z0
          x1, z1 = xint, zint
          T = -(1/l)*math.log(1-get_random_uniform(rng_states))
          internal_state[0] = uint32(T)

      # Periodic boundary condition along x:
      x1 = (L/2 + x1) % (L) - L/2

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()


  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)
    L, h = self.L, self.h
    r0 = rng.uniform((-L/2, -h/2), (L/2, h/2), size=(N, 2))
    return r0.astype(dtype)

  def plot(self, ax=None, border_kwargs={'c': 'r'}):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    L, h = self.L, self.h

    # Draw geometry

    ax.plot([-L/2, +L/2], [-h/2, -h/2], **border_kwargs)
    ax.plot([-L/2, +L/2], [+h/2, +h/2], **border_kwargs)

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax


class Periodic(Topology):
  __version__ = '0.0.2'

  def __init__(self, L: dtype, **kwargs) -> None:
    """Just periodic box without any walls

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
    """
    self.L = L
    regions = []
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      # Periodic condition
      x1 = (L/2 + x1) % (L) - L/2
      z1 = (L/2 + z1) % (L) - L/2

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()


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
  __version__ = '0.0.4'

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
    """
    self.Lm = Lm
    self.L = L
    self.R = R
    regions = [{'name': 'left', 'def': 'x<=0'}]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      # Intersection with left wall
      X, Z = -L/2, 0
      NX, NZ = 1, 0
      if x1 < X:
        x1 = x1+2*(X-x1)

      # Intersection with membrane
      X, Z = 0, 0
      NX, NZ = 1, 0
      if x0*x1 <= 0 and x1 != x0:
        t = (X-x0)/(x1-x0)
        zint = t*z1 + (1-t)*z0
        if math.fabs(zint) > R:
          x1 *= -1

      # Intersection with right wall
      X, Z = +L/2, 0
      NX, NZ = 1, 0
      if x1 > X:
        x1 = x1+2*(X-x1)

      z1 = (Lm + z1) % (2*Lm) - Lm

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

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
    ax.plot([L/2,  L/2], [-Lm, +Lm], **border_kwargs)

    ax.plot([0, 0], [R, Lm], **border_kwargs)
    ax.plot([0, 0], [-R, -Lm], **border_kwargs)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax


class ElasticChannel1(Topology):
  __version__ = '0.0.5'

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
    """
    # TODO : Use Marbach notation ?
    self.L = L
    self.h = h
    self.R = R
    regions = [{'name': 'left', 'def': 'x<=0'}]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2

        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1) < R+L/2) and (math.fabs(x1) > L/2) and \
           (math.fabs(x0) < R+L/2) and (math.fabs(x0) > L/2):
          break

        # Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1 < X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0 < X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if z0 > Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if z0 < Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if x0 > X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if x1 > X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)

      # Periodic boundary condition along z:
      z1 = (R + z1) % (2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

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
    ax.plot([+L/2+R,  L/2+R], [-R, +R], **border_kwargs)

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


class ElasticChannel2(Topology):
  __version__ = '0.0.3'

  def __init__(self,
               L: dtype, H: dtype,
               Lc: dtype, Hc: dtype, **kwargs) -> None:
    """Create a new channel geometry

    ┃         ┃   ┃         ┃      ↑  
    ┃         ┃   ┃         ┃      │
    ┃         ┗━━━┛         ┃      │
    ┃                       ┃ ↕ Hc │ 2 H
    ┃         ┏━━━┓         ┃      │
    ┃         ┃   ┃         ┃      │
    ┃         ┃   ┃         ┃      ↓
     ←-------→ ←-→ ←-------→
         L      Lc     L


    ┃ : Elastic wall

    Args:
      L (float in A): Depth of reservoirs
      H (float in A): Reservoir height
      Lc (float in A): Length of the channel
      Hc (float in A): Height of the channel
    """
    self.L, self.H = L, H
    self.Lc, self.Hc = Lc, Hc
    regions = [{'name': 'left',  'def': 'x<=-Lc/2'},
               {'name': 'right', 'def': 'x>=Lc/2'}]
    # regions=[]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2

        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1) < L+Lc/2) and (math.fabs(x1) > Lc/2) and \
           (math.fabs(x0) < L+Lc/2) and (math.fabs(x0) > Lc/2):
          break

        # Intersection with left wall
        X, Z = -L-Lc/2, 0
        NX, NZ = 1, 0
        if x1 < X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -Lc/2, 0
        NX, NZ = 1, 0
        if x0 < X and x1 > X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > Hc/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             X, Z,
                                             NX, NZ)
                break

        # Intersection with bottom channel
        X, Z = 0, -Hc/2
        NX, NZ = 0, 1
        if z0 > Z and z1 < Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < Lc/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             X, Z,
                                             NX, NZ)
                toCheck = True
                continue

        # Intersection with top channel
        X, Z = 0, +Hc/2
        NX, NZ = 0, 1
        if z0 < Z and z1 > Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < Lc/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             X, Z,
                                             NX, NZ)
                toCheck = True
                continue

        # Intersection with right membrane
        X, Z = +Lc/2, 0
        NX, NZ = 1, 0
        if x0 > X and x1 < X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(zint) > Hc/2:
                x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                             x1, z1,
                                             X, Z,
                                             NX, NZ)
                break

        # Intersection with right wall
        X, Z = +L+Lc/2, 0
        NX, NZ = 1, 0
        if x1 > X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)

      # Periodic boundary condition along z:
      z1 = (H + z1) % (2*H) - H

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L, H = self.L, self.H
    Lc, Hc = self.Lc, self.Hc

    # Surface of reservoirs
    S_R = (2*L) * (2*H)
    # Surface of the channel
    S_c = Lc * Hc

    # Put particles in reservoirs
    N_R = int(np.ceil(N*S_R/(S_R+S_c)))
    r0_R = rng.uniform((-L, -H), (L, H), size=(N_R, 2))

    r0_R[np.where(r0_R[:, 0] < 0), 0] -= Lc/2
    r0_R[np.where(r0_R[:, 0] >= 0), 0] += Lc/2

    # Number of particles in channel
    N_c = N-N_R
    if N_c > 0:
      r0_c = rng.uniform((-Lc/2, -Hc/2), (Lc/2, Hc/2), size=(N_c, 2))
      r0 = np.concatenate((r0_R, r0_c))
    else:
      r0 = r0_R

    return r0.astype(dtype)

  def plot(self, ax=None,
           border_kwargs={'elastic': {'c': 'r'},
                          'periodic': {'c': 'r', 'ls': '--'}
                          }):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    # Get geometry parameters
    L, H = self.L, self.H
    Lc, Hc = self.Lc, self.Hc

    # Draw geometry
    # Reservoir borders
    ax.plot([-L-Lc/2, -L-Lc/2], [-H, +H], **border_kwargs['elastic'])
    ax.plot([+L+Lc/2, +L+Lc/2], [-H, +H], **border_kwargs['elastic'])
    # Periodic boundary condition
    ax.plot([-L-Lc/2, -Lc/2], [-H, -H], **border_kwargs['periodic'])
    ax.plot([-L-Lc/2, -Lc/2], [+H, +H], **border_kwargs['periodic'])
    ax.plot([+L+Lc/2, +Lc/2], [-H, -H], **border_kwargs['periodic'])
    ax.plot([+L+Lc/2, +Lc/2], [+H, +H], **border_kwargs['periodic'])
    # Membranes
    ax.plot([-Lc/2, -Lc/2], [-H, -Hc/2], **border_kwargs['elastic'])
    ax.plot([-Lc/2, -Lc/2], [+H, +Hc/2], **border_kwargs['elastic'])
    ax.plot([+Lc/2, +Lc/2], [-H, -Hc/2], **border_kwargs['elastic'])
    ax.plot([+Lc/2, +Lc/2], [+H, +Hc/2], **border_kwargs['elastic'])
    # Channel wall
    ax.plot([-Lc/2, +Lc/2], [-Hc/2, -Hc/2], **border_kwargs['elastic'])
    ax.plot([-Lc/2, +Lc/2], [+Hc/2, +Hc/2], **border_kwargs['elastic'])

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax


class AbsorbingChannel1(Topology):
  __version__ = '0.0.5'

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
    """

    self.L = L
    self.h = h
    self.R = R
    self.l = l
    regions = [{'name': 'left', 'def': 'x<=0'},
               {'name': 'absorbed', 'def': 'internal_state[0] > 0'}]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @cuda.jit(device=True)
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
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
        if (math.fabs(x1) < R+L/2) and (math.fabs(x1) > L/2) and \
           (math.fabs(x0) < R+L/2) and (math.fabs(x0) > L/2):
          break

        # Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1 < X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0 < X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if z0 > Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = xint, zint+1
                T = -(1/l)*math.log(1-get_random_uniform(rng_states))
                internal_state[0] = uint32(T)
                break

        # Intersection with top channel
        X, Z = 0, +h/2
        NX, NZ = 0, 1
        if z0 < Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
              xint = t*x1 + (1-t)*x0
              zint = t*z1 + (1-t)*z0
              if math.fabs(xint) < L/2:
                x1, z1 = xint, zint-1
                T = -(1/l)*math.log(1-get_random_uniform(rng_states))
                internal_state[0] = uint32(T)
                break

        # Intersection with right membrane
        X, Z = +L/2, 0
        NX, NZ = 1, 0
        if x0 > X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if x1 > X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)

      # Periodic boundary condition along z:
      z1 = (R + z1) % (2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

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
    ax.plot([+L/2+R,  L/2+R], [-R, +R], **border_kwargs)

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
  __version__ = '0.0.2'

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
    """
    # TODO : Use Marbach notation ?
    self.L = L
    self.h = h
    self.R = R
    regions = []
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      toCheck = True
      i_BOUNCE = 0
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          x1 = (x0+x1)/2
          z1 = (z0+z1)/2

        # Fast skip if trajectory stay in reservoirs
        if (math.fabs(x1) < R+L/2) and (math.fabs(x1) > L/2) and \
           (math.fabs(x0) < R+L/2) and (math.fabs(x0) > L/2):
          break

        # Intersection with left wall
        X, Z = -R-L/2, 0
        NX, NZ = 1, 0
        if x1 < X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        # Intersection with left membrane
        X, Z = -L/2, 0
        NX, NZ = 1, 0
        if x0 < X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if z0 > Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if z0 < Z:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if x0 > X:
          den = (x1-x0)*NX + (z1-z0)*NZ
          if den != 0:
            t = ((X-x0)*NX + (Z-z0)*NZ)/den
            if (t > 0) and (t < 1):
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
        if x1 > X:
          x1, z1 = bc.ReflectIntoPlane(x0, z0,
                                       x1, z1,
                                       X, Z,
                                       NX, NZ)
          break

        i_BOUNCE += 1
        # if i_BOUNCE>=MAX_BOUNCE-2:
        #   print(pos, i_BOUNCE, x0, z0, x1, z1)

      # Periodic boundary condition along z:
      z1 = (R + z1) % (2*R) - R

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

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
    ax.plot([+L/2+R,  L/2+R], [-R, +R], **border_kwargs)

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

class TBElasticChannel(Topology):
  __version__ = '0.0.1'
  def __init__(self,
               L: dtype, H: dtype,
               Lc: dtype, Hc: dtype, Rc:dtype, 
               **kwargs) -> None:
    """Create a new top-bottom channel with elastic boundary condition

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━   
                 2Lc     top          ↑
           ←------------→             │
           ━━━━━━━━━━━━━━       ↑ Hc  │    
    ━━━━━━━━━━━━    ━━━━━━━━━━━━↓     │ 2H
                ←--→                  │
                 Rc                   │
                          bottom      ↓
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━   
    ←--------------------------→
                 2L
      ---→
    │    z
    ↓ x        


    ┃ : Elastic wall

    Args:
      L (float in A): Depth of reservoirs
      H (float in A): Reservoir height
      Lc (float in A): Length of the channel
      Hc (float in A): Height of the channel
      Rc (float in A): Exit size of the channel
    """
    self.L, self.H = L, H
    self.Lc, self.Hc, self.Rc = Lc, Hc, Rc
    regions = [
               {'name': 'inside','def': 'x<=0 and x>=-Hc and math.fabs(z)<=Lc'},
               {'name': 'after', 'def': 'x>=0'}]
    # regions=[]
    self.regions = regions

    # Geometrical parameters are treated as constants during the compilation
    @jit
    def compute_boundary_condition(x0: dtype, z0: dtype,
                                   x1: dtype, z1: dtype,
                                   rng_states: array,
                                   internal_state: tuple):
      toCheck = True
      i_BOUNCE = 0
      r0 = (x0, z0)
      r1 = (x1, z1)
      while toCheck and i_BOUNCE < MAX_BOUNCE:
        toCheck = False
        if i_BOUNCE > 4:
          r1 = vector2D.mean(r0, r1)

        # Fast skip
        # Bottom reservoir
        if r0[0]<=-Hc and r1[0]<=-Hc and r1[0]>=-H : break
        # Top reservoir
        if r0[0]>=0 and r1[0]>=0 and r1[0]<=H : break
        # Inside channel
        if r0[0]>=-Hc and r0[0]<=0 and \
           r1[0]>=-Hc and r1[0]<=0 : break

        # Intersection with top wall
        r1, doCollide = vector2D.get_reflected(r0, r1, 
                                                (-H, -5*L), (-H, +5*L))
        if doCollide: break
        # Intersection with bottom wall
        r1, doCollide = vector2D.get_reflected(r0, r1, 
                                                (+H, -5*L), (+H, +5*L))
        if doCollide: break

        # Intersection with top layer
        r1, doCollide = vector2D.get_reflected(r0, r1, 
                                                (-Hc, -Lc), (-Hc, +Lc))
        if doCollide: toCheck=True

        # Intersection with bottom left layer
        r1, doCollide = vector2D.get_reflected(r0, r1, 
                                                (0, -5*L), (0, -Rc/2))
        if doCollide: toCheck=True

        # Intersection with bottom right layer
        r1, doCollide = vector2D.get_reflected(r0, r1, 
                                                (0, Rc/2), (0, +5*L))
        if doCollide: toCheck=True

        i_BOUNCE += 1

      x1, z1 = r1

      # Periodic boundary condition along z:
      z1 = (L + z1) % (2*L) - L

      return x1, z1
    self.compute_boundary_condition = compute_boundary_condition

    super().__init__()

  def fill_geometry(self, N: uint, seed=None) -> ndarray:
    rng = np.random.default_rng(seed)

    # Get geometry parameters
    L, H = self.L, self.H
  
    # Put particles
    r0 = rng.uniform((-L, -H), (L, H), size=(N, 2))

    return r0.astype(dtype)

  def plot(self, ax=None,
           border_kwargs={'elastic': {'c': 'r'},
                          'periodic': {'c': 'r', 'ls': '--'}
                          }):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.get_figure()

    # Get geometry parameters
    L, H = self.L, self.H
    Lc, Hc, Rc = self.Lc, self.Hc, self.Rc

    # Draw geometry
    # Reservoir borders
    ax.plot([-H, -H], [-L, +L], **border_kwargs['elastic'])
    ax.plot([+H, +H], [-L, +L], **border_kwargs['elastic'])
    # Periodic boundary condition
    ax.plot([-H, +H], [-L, -L], **border_kwargs['periodic'])
    ax.plot([-H, +H], [+L, +L], **border_kwargs['periodic'])
    # Top layer
    ax.plot([-Hc, -Hc], [-Lc, +Lc], **border_kwargs['elastic'])
    ax.plot([0, 0], [-L, -Rc/2], **border_kwargs['elastic'])
    ax.plot([0, 0], [Rc/2, L], **border_kwargs['elastic'])

    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    return fig, ax