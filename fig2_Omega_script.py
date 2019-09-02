from __future__ import division, print_function

import os
import numpy as np
import math
from math import pi
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from library import *

if __name__ == '__main__':
    
    # Directory and files to import from
    folder_Omega = 'Fig2_dataset'
    file_list = ['delta0.mat', 'delta1.mat', 'delta2.mat']
    color_list = ['black', 'blue', 'green']
    
    dir_Omega = os.path.join(os.getcwd(), 'data', folder_Omega)
    
    
    # Figure
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3d plot for each file in file_list:
    for i in range(len(file_list)):
        Omega_file = file_list[i]
        
        # Import meshes
        Omega_mat = sio.loadmat(os.path.join(dir_Omega, Omega_file))
        
        Omega_mesh = Omega_mat['Omega']
        gain_mesh = Omega_mat['gain']
        T_mesh = Omega_mat['T']
        
        # Plot surface
        ax.plot_wireframe(T_mesh, gain_mesh, Omega_mesh, color=color_list[i])
    
    # Plot w0:
    w0 = Omega_mat['w0']
    w0_mesh = w0*np.ones(T_mesh.shape)
    ax.plot_surface(T_mesh, gain_mesh, w0_mesh, color='grey', alpha=0.5)
    
    # Figure options
    ax.set_zlim(0, w0)    