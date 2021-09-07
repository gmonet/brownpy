from numba import cuda
_GPU_COMPUTATION = cuda.is_available()

def set_computation_type(type='auto'):
    global _GPU_COMPUTATION
    if type=='auto':
        _GPU_COMPUTATION = cuda.is_available()
    elif type=='cpu':
        _GPU_COMPUTATION = False
    elif type=='gpu':
        _GPU_COMPUTATION = True
    else:
        raise ValueError("type should be \'auto\', \'cpu\' or \'gpu\'")

