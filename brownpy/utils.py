import math
import numpy as np
from numba import cuda

def prefix(x, dimension=1):
  """Give the number an appropriate SI prefix.

  :param x: Too big or too small number.
  :returns: String containing a number between 1 and 1000 and SI prefix.
  https://stackoverflow.com/questions/29627796/pretty-printing-physical-quantities-with-automatic-scaling-of-si-prefixes
  """
  if x == 0:
    return "0  "

  l = math.floor(math.log10(abs(x)))

  div, mod = divmod(l, 3*dimension)
  return "%.3g %s" % (x * 10**(-l + mod), " kMGTPEZYyzafpnµm"[div])


def unwrap(x, period):
  shiftx = np.zeros_like(x)
  shiftx[np.where(np.diff(x) > +period/2)[0]+1] -= period
  shiftx[np.where(np.diff(x) < -period/2)[0]+1] += period
  shiftx = np.cumsum(shiftx)
  return x+shiftx

@cuda.jit
def _reset_inside(inside):
  pos = cuda.grid(1)
  if pos<inside.shape[1]:
      for i in range(inside.shape[0]):
          inside[i, pos] = 0
def reset_inside(inside):
  '''
  Very simple function that reset value of inside array to 0 staying inside device 
  '''
  threadsperblock = 32
  blockspergrid = math.ceil(inside.shape[1]/threadsperblock)
  _reset_inside[blockspergrid, threadsperblock](inside)