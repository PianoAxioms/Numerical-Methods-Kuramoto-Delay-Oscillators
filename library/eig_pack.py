from __future__ import division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi

from scipy import integrate
from scipy import optimize

# ALL FUNCTIONS FOR ROOT FINDING AND EIGENVALUES

# Root finding for one-dimensional f(x) = 0, closest to initial x_0

class re_root:
    '''
    A class containing a status and returned value, as well as the function arguments.
    '''
    
    def __init__(self):
        
        self.status = 'None'
        self.x = 0
        
        self.f = None
        self.interval = None
        self.part = None
        
        
def closest_root(f, x0, interval, part=20):
    '''
    Finds a root of func with initial guess x0. Utilizes the optimize.root and optimize.bisect functions. If multiple roots are found, returns
    the closest root to x0. part is the partition number N0 in which the array of f is computed over a,b.
    '''
    
    # Output class
    root = re_root()

    root.f = f
    root.interval = interval
    root.part = part
    
    # Find all roots
    all_roots = roots_in_interval(f, interval, part=part)
    
    if all_roots.size == 0:
        root.status = 'None'
    
    else:
        root.status = 'Exists'
        root.x = all_roots[np.argmin(np.abs(all_roots - x0))]
    
    return root


def roots_in_interval(f, interval, part=20):
    '''
    Given a 1-D function func, finds all roots of the function on the interval [a,b]. part is the partition number N0 in which the array of f is
    computed over a,b. Utilises the optimize.bisect function.
    '''
    
    [a,b] = interval
    N0 = part
    
    # Compute all arrays
    x_array = np.linspace(a,b, num=N0)
    f_array = np.array([f(x_array[k]) for k in range(N0)])
    f_array_L = f_array[:-1]
    f_array_R = f_array[1:]
    
    # Initial zero check:
    zero_array = x_array[f_array == 0]
    
    # Check for all sign changes:
    sign_array = np.argwhere(f_array_L*f_array_R < 0)
    
    # Implement bisect method:
    root_array = np.array([optimize.bisect(f, x_array[l], x_array[l+1]) for l in sign_array])
    all_root_array = np.concatenate((zero_array, root_array))
    
    return all_root_array


# Root finding for two-dimensional (complex) f(z) = 0 in a rectangular region R.

class ImRoot:
    '''
    A class holding all information regarding complex root (eigenvalue) searching.
    Can iterate using the subregion function.
    '''
    
    def __init__(self):
        
        self.rect = np.zeros((2,2))
        self.err = 0
        self.parts = [0,0]
        self.f = lambda z: 0        
        
        # For 1D partitions
        self.u_part = np.array([])
        self.v_part = np.array([])
        
        # 2D arrays
        self.u_mesh = np.array([])
        self.v_mesh = np.array([])
        self.f_mesh = np.array([])
        self.z_mesh = np.array([])
        

class ImAllRoot(ImRoot):
    '''
    A class managing the refined arrays in the entire region.
    '''
    
    def __init__(self):
        
        ImRoot.__init__(self)
        
        # Refined 1D partitions
        self.u_refpart = np.array([])
        self.v_refpart = np.array([])
        
        # Refined 2D meshes
        self.u_refmesh = np.array([])
        self.v_refmesh = np.array([])
        self.f_refmesh = np.array([])
        
        # Collection of regions where a zero might be.
        self.cover = []        
        

class ImSubRoot(ImRoot):
    '''
    A class managing the arrays in an isolated region.
    '''
    
    def __init__(self):
        
        ImRoot.__init__(self)
        
        # Minimum point
        self.min_f = 0
        self.min_point = (0,0)
        
        # Center of mass (using Gaussian weights)
        self.center = (0,0)
        
        
def root_mesh(f, rect, parts=(50,50), root='all'):
    '''
    Returns an im_root with the corresponding arrays partitioning rect (2x2 array).
    The arrays are the u-mesh, v-mesh, and f(u+iv)-mesh. f is a complex function.
    '''
    
    # Create im_root instance
    if root == 'all':
        root = ImAllRoot()
    else:
        root = ImSubRoot()
    
    # Obtain mesh of region
    u_part = np.linspace(rect[0,0], rect[0,1], parts[0])
    v_part = np.linspace(rect[1,0], rect[1,1], parts[1])
    v_mesh, u_mesh = np.meshgrid(v_part, u_part)

    
    # Compute every value of |f|
    f_mesh = np.zeros(parts, dtype=complex)
    
    for k in range(parts[0]):
        for l in range(parts[1]):
            f_mesh[k,l] = f(u_mesh[k,l] + v_mesh[k,l]*1j)
    
    # Update root
    root.rect = rect
    root.f = f
    
    root.u_part = u_part
    root.v_part = v_part
    
    root.u_mesh = u_mesh
    root.v_mesh = v_mesh
    root.f_mesh = f_mesh
    
    return root

    
def subregion(f, rect, err, parts=(50,50)):
    '''
    Returns an im_root with a list of rectangles as a 2x2 array: [[Re_L, Re_R, Im_L, Im_R]] such that |f| < err. Here, rect
    is the rectangle on which all f values are computed, and parts is the partition size along the re and im axes.
    '''
    
    root = root_mesh(f, rect, parts=parts, root='all')
    
    u_part = root.u_part
    v_part = root.v_part
    f_mesh = root.f_mesh
    
    # Compute Gaussian error z
    z_mesh = np.exp(-np.abs(f_mesh)**2/2)
    root.z_mesh = z_mesh
    
    # Get logistic f_mesh for |f| < err
    f_bool = np.abs(f_mesh) < err
    f_int = f_bool.astype(int)

    cover_list = []
    while np.count_nonzero(f_int) > 0:
        
        # Start at the maximum point (smallest error)
        p = np.where(z_mesh*f_int == np.max(z_mesh*f_int))
        p = (p[0][0], p[1][0])
        
        # Obtain rectangle
        R = bounding_rectangle(f_bool, p)
    
        # Indices in R
        R_x, R_y = bounded_indices(R)
    
        # Remove the covering region R on f_bool
        f_bool[R_x, R_y] = False
        f_int = f_bool.astype(int)
        
        # Add to cover
        cover_list.append(R)
    

    # Convert each covering region into Cartesian set
    cover_set_list = []
    for R in cover_list:
        
        subrect = np.zeros((2,2), dtype='float64')
        subrect[0,0] = u_part[R[0,0]]
        subrect[0,1] = u_part[R[0,1]]
        subrect[1,0] = v_part[R[1,0]]
        subrect[1,1] = v_part[R[1,1]]
        
        cover_set_list.append(subrect)
    
    # Update root
    root.cover = cover_set_list
    
    return root


def get_center(f, rect, parts=(50,50)):
    '''
    Given a region where there may be a point f(z) = 0, returns a root class with the Gaussian center of mass, 
    given by integrating e^{-|z|^2/2} for all errors. Any point where |z| > tthold is not counted and is set to
    zero. Also returns the point with the minimum |f| value, along with its value.
    '''
    
    root = root_mesh(f, rect, parts=parts, root='sub')
    
    u_mesh = root.u_mesh
    v_mesh = root.v_mesh
    f_mesh = root.f_mesh
    
    # Compute Gaussian error z
    z_mesh = np.exp(-np.abs(f_mesh)**2/2)
    root.z_mesh = z_mesh
    
    # Minimum value:
    min_point = np.where(z_mesh == np.max(z_mesh))
    min_inds = (min_point[0][0], min_point[1][0])
    
    root.min_point = (u_mesh[min_inds], v_mesh[min_inds])
    root.min_f = f_mesh[min_inds[0], min_inds[1]]
    
    # Update z_mesh:
    z_mesh = z_mesh
    
    # Center of mass
    z_mass = np.sum(z_mesh)
    X = np.sum(u_mesh*z_mesh) / z_mass
    Y = np.sum(v_mesh*z_mesh) / z_mass

    # Store the center of mass
    root.center = (X,Y)
    
    return root


def integrate_subroot(all_root, sub_root):
    '''
    Given an AllRoot and SubRoot class on which the SubRoot.rect is contained in AllRoot.rect and the
    f_mesh agrees, loads a mesh onto all_root's refmesh that is a refinement of mesh where each submesh is 
    contained in mesh. Linear interpolation is applied to compute other points.
    '''
    
    pass


# SUPPLEMENTARY FUNCTIONS

    
def bounding_rectangle(bool_mesh, p):
    '''
    Given a point p = (i,j) in bool_mesh (a logistic mesh), returns a bounding subregion (2x2 array) by moving in the 
    left, right, up, down directions from p. For each direction, we move until a False point is found or until we 
    hit the edge of mesh_bool.
    '''
    
    (row, col) = bool_mesh.shape
    (i,j) = p
    
    R = np.zeros((2,2))
    
    # Left direction
    k = i
    q_bool = True
    while q_bool and k > 0:
        k -= 1
        q_bool = bool_mesh[k, j]
    
    R[0,0] = k
    
    # Right direction
    k = i
    q_bool = True
    while q_bool and k < row-1:
        k += 1
        q_bool = bool_mesh[k, j]    
        
    R[0,1] = k
    
    # Down direction
    k = j
    q_bool = True
    while q_bool and k > 0:
        k -= 1
        q_bool = bool_mesh[i, k]  
    
    R[1,0] = k
    
    # Up direction
    k = j
    q_bool = True
    while q_bool and k < col-1:
        k += 1
        q_bool = bool_mesh[i, k]  
    
    R[1,1] = k
    
    # Convert to integer
    R = R.astype(int)
    
    return R


def bounded_indices(rect):
    '''
    Given a rect that is a 2x2 array of [[ind_L, ind_R], [ind_D, ind_U]] of integer values, returns all integers 
    in rect with the region bounds, as two arrays (x,y). Then, for any 2-dim array A, one can assign values by A[x,y].
    '''
    
    xL = rect[0,0]
    xR = rect[0,1]
    yL = rect[1,0]
    yR = rect[1,1]
    
    x_inds = np.arange(xL, xR)
    y_inds = np.arange(yL, yR)
    
    x_array = np.tile(x_inds, len(y_inds))
    y_array = np.repeat(y_inds, len(x_inds))
    
    # Convert to integer
    x_array = x_array.astype(int)
    y_array = y_array.astype(int)
    
    return x_array, y_array


# Test plots
def heatmap(ax, u_mesh, v_mesh, z_mesh):
    '''
    Plots the heatmap using the u, v, z meshes. Assumes that z is non-negative.
    '''
    
    z_max = np.max(z_mesh)
    
    u_min = -2 # np.min(u_mesh)
    u_max = 2 # np.max(u_mesh)
    v_min = -2 # np.min(v_mesh)
    v_max = 2 # np.max(v_mesh)
    
    ax.axis([u_min, u_max, v_min, v_max])
    
    c = ax.pcolormesh(u_mesh, v_mesh, z_mesh, cmap='binary', vmin=0, vmax=z_max) 
    

def contour(ax, u_mesh, v_mesh, z_mesh, thhold=0.1, alpha=1):
    '''
    Plots a contour map using the u, v, z meshes. Asusmes that z has a range from 0 to 1,
    with 1 being the colour and 0 being white.
    '''
    
    levels = np.linspace(thhold, 1, 9)
    levels = np.concatenate((np.array([0]), levels))
    ax.contourf(u_mesh, v_mesh, z_mesh, levels, cmap='Blues', alpha=alpha)
    
    
if __name__ == '__main__':
    region = np.array([[-5,5], [-5,5]])
    mesh = (100,100)
    
    f = lambda z: z**2 + 1
    err = 0.2
    
    R = subregion(f, region, err, mesh)
    Q = get_center(f, region)
    
    fig, ax = plt.subplots(1,1)
    
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    
    contour(ax, R.u_mesh, R.v_mesh, R.z_mesh, thhold=0.1, alpha=0.8)
    
        # ax.scatter(np.array([S.center[0]]), np.array([S.center[1]]), color='black')
        
    