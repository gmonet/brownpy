import cupy as cp
import numpy as np
from numba import jit
import numba as nb

def PSD_gpu(x, dt):
    N = len(x)
    x_gpu = cp.asarray(x)
    psd = (cp.abs(cp.fft.rfft(x_gpu))**2/N).get()*dt
    freqs = (cp.fft.rfftfreq(x.size, dt)).get()
    del x_gpu
    cp._default_memory_pool.free_all_blocks()
    return psd, freqs

def PSD_gpu_binned(x, dt, bins):
    N = len(x)
    x_gpu = cp.asarray(x)
    psd_gpu = cp.abs(cp.fft.rfft(x_gpu))**2/N*dt
    freqs_gpu = cp.fft.rfftfreq(x.size, dt)

    psd_binned_gpu, freqs_binned_gpu = cp.histogram(cp.log(freqs_gpu)[1:], 
                                        weights=psd_gpu[1:], bins=bins)
    N_binned_gpu = cp.histogram(cp.log(freqs_gpu)[1:], bins=bins)[0]
    psd_averaged = (psd_binned_gpu/N_binned_gpu).get()
    freqs = ((freqs_binned_gpu[1:]+freqs_binned_gpu[:-1])/2.).get()
    
    del x_gpu, psd_gpu, freqs_gpu, psd_binned_gpu, freqs_binned_gpu, N_binned_gpu
    cp._default_memory_pool.free_all_blocks()
    return psd_averaged, np.exp(freqs)

# https://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
# https://www.dfki.de/fileadmin/user_upload/import/7051_131008_Memory_and_Processing_Efficient_Formula_for_Moving_Variance_Calculation_in_EEG_and_EMG_Signal_Processing_NEUROTECHNIX_Krell.pdf

@jit(nopython=True)
def _compute_MRV(x, n, ddof=0):
    '''
    Compute Mean Rolling Variance (MRV)

    Attributes:
        x (ndarray): Description of `attr1`.
        n (int): Window size
        ddof(int, optional): Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.
    '''
    N = len(x)
    if n > N:
        return np.var(x)
    
    div = n-ddof
    mn = np.sum(x[0:n])
    Sn = n**2*np.var(x[0:n])
    sum_Sn = Sn
    for i_0 in range(N-n):
        i_n = i_0 + n
        i_n = i_0 + n
        x_0, x_n = x[i_0], x[i_n]
        Sn = Sn + (x_n-x_0)*((n-1)*x_n + (n+1)*x_0 - 2*mn)
        mn = mn + (x_n-x_0)
        sum_Sn += Sn
    return sum_Sn/n/div/(N-n+1)

@jit(nopython=True, parallel=True)
def compute_MRV(x, windows, ddof=0):
    '''
    Compute Mean Rolling Variance (MRV)

    Attributes:
        x (ndarray): Description of `attr1`.
        intervals (ndarray): List of window size
        ddof(int, optional): Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.

    '''
    N_int = len(windows)
    result = np.empty(N_int)
    for i in nb.prange(N_int):
        result[i] = _compute_MRV(x, windows[i], ddof)
    
    return result