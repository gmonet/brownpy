from numba import cuda

def set_computation_type(type='auto'):
    if type=='auto':
        _GPU_COMPUTATION = cuda.is_available()
    elif type=='cpu':
        _GPU_COMPUTATION = False
    elif type=='gpu':
        _GPU_COMPUTATION = True
    else:
        raise ValueError("type should be \'auto\', \'cpu\' or \'gpu\'")
    return _GPU_COMPUTATION

_GPU_COMPUTATION = set_computation_type()