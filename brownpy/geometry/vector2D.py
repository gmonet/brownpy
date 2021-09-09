from numba import jit
import math

@jit(inline='always')
def det(a, b):
    return a[0]*b[1]-a[1]*b[0]
@jit(inline='always')
def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]
@jit(inline='always')
def add(a, b):
    return (a[0]+b[0], a[1]+b[1])
@jit(inline='always')
def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])
@jit(inline='always')
def edot(l, a):
    return (l*a[0], l*a[1])
@jit(inline='always')
def norm(a):
    return math.sqrt(a[0]**2+a[1]**2)
@jit(inline='always')
def normed_perp(a):
    n = norm(a)
    return (a[1]/n, -a[0]/n) 
# class Point2D:
#     @jit
#     def cross(a, b):
#         return a[0]*b[1]-a[1]*b[0]
#     @jit
#     def dot(a, b):
#         return a[0]*b[0]+a[1]*b[1]
# Vector2D
@jit(inline='always')
def det(a, b):
    return a[0]*b[1]-a[1]*b[0]
@jit(inline='always')
def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]
@jit(inline='always')
def add(a, b):
    return (a[0]+b[0], a[1]+b[1])
@jit(inline='always')
def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])
@jit(inline='always')
def edot(l, a):
    return (l*a[0], l*a[1])
@jit(inline='always')
def norm(a):
    return math.sqrt(a[0]**2+a[1]**2)
@jit(inline='always')
def mean(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)
@jit(inline='always')
def normed_perp(a):
    n = norm(a)
    return (a[1]/n, -a[0]/n) 
@jit
def get_reflected(p0, p1, W0, W1):
    '''
    Get reflected point onto a segment. 
          p0  W0
           ╲  │
            ╲ │
             ╲│
             ╱│
            ╱ │╲
           ╱  │ ╲
          ╱   │  ╲
        p1p   W1  p1
    args:
        p0 : starting point
        p1 : end point
        W0 : starting wall point
        W1 : end wall point
    return:
        p1p: reflected point if trajectory cross wall else p1
        doCross (bool)
    '''
    dW = sub(W1, W0)
    dp = sub(p1, p0)
    d = det(dp, dW)
    if d== 0: # colinear
        return ((p1[0], p1[1]), False)
    t = det(sub(W0, p0), dW)/d
    u = det(sub(W0, p0), dp)/d
    if t>1 or t<0 or u>1 or u<0 : 
        return ((p1[0], p1[1]), False)
    C = add(p0, edot(t,dp)) # Crossing point
    # Find reflected point
    outer_dp = sub(p1, C) # the remaining trajectory after crossing wall
    nW = normed_perp(dW) # Unitary normal vector of the wall
    p1p = add(C, sub(outer_dp, edot(2*dot(outer_dp,nW),nW)))
    return (p1p, True)

if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    #Some testing
    N = 21
    fig, axes =plt.subplots(N, figsize=(5,N*3))
    p0 = (-1, 0)
    p1 = (2, -0.50)
    for i, theta in enumerate(np.linspace(0, np.pi, N)):
        ax = axes[i]
        W0 = (np.cos(theta), np.sin(theta))
        W1 = (2+-2*np.cos(theta), -2*np.sin(theta))
        
        p1p = get_reflected(p0, p1, W0, W1)
        dW = sub(W1, W0)
        dp = sub(p1, p0)
        d = det(dp, dW)
        if d== 0: 
            print('colinear')
            p1p = p1
            # return p1
        else:
            t = det(sub(W0, p0), dW)/d
            u = det(sub(W0, p0), dp)/d
            C = add(p0, edot(t,dp))
            outer_dp = sub(p1, C)
            nW = normed_perp(dW)
            if t>1 or t<0 or u>1 or u<0 : 
                print('do not cross')
                p1p = p1
                # return p1
            else:
                # C = add(p0, edot(t,dp))
                # Find reflected point
                p1p = add(C,sub(outer_dp, edot(2*dot(outer_dp,nW),nW)))
                p1p, _ = get_reflected(p0, p1, W0, W1)
                ax.plot([C[0], p1p[0]], [C[1], p1p[1]], c='b', lw=3, ls=':')

        ax.plot([W0[0], W1[0]], [W0[1], W1[1]], c='k', lw=3)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c='b', lw=1)
        # plot_vector(C, outer_dp)
        # plot_vector(C, nW)
        ax.scatter([C[0]], [C[1]], c='k', lw=1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    
    from numba import cuda
    @cuda.jit
    def test(array_p0, array_p1, W0, W1, array_p1p):
        pos = cuda.grid(1)
        if pos<array_p0.shape[0]:
            p1p, _ = get_reflected(array_p0[pos], array_p1[pos], W0, W1)
            array_p1p[pos, 0] = p1p[0]
            array_p1p[pos, 1] = p1p[1]
            
    array_p0 = np.random.uniform(-2, 2, size=(1024, 2))
    array_p1 = np.random.uniform(-2, 2, size=(1024, 2))
    array_out = np.empty(shape=(1024, 2))
    threadsperblock = 32
    blockspergrid = math.ceil(array_p0.shape[0]/ threadsperblock)
    N=10
    fig, axes =plt.subplots(N, figsize=(5,N*3))
    for i, theta in enumerate(np.linspace(0, np.pi, N)):
        ax = axes[i]
        W0 = (np.cos(theta), np.sin(theta))
        W1 = (2+-2*np.cos(theta), -2*np.sin(theta)) 
        test[blockspergrid, threadsperblock](array_p0, array_p1, W0, W1, array_out)
        ax.plot([W0[0], W1[0]], [W0[1], W1[1]], c='k', lw=3)
        for i in range(10):
            c=f'C{i}'
            p0, p1 = array_p0[i], array_p1[i]
            p1p = array_out[i]
            
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c=c, lw=1)
            ax.scatter([p1p[0]], [p1p[1]], c=c)
        # plot_vector(C, outer_dp)
        # plot_vector(C, nW)
        # ax.scatter([C[0]], [C[1]], c='k', lw=1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        
        