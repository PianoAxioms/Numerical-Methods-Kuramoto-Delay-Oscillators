from __future__ import division, print_function

import os

import matplotlib.pyplot as plt
import matplotlib.transforms as tsfm
import matplotlib.font_manager as fnt
import matplotlib.lines as lnes

import numpy as np
from math import pi


# PROVIDES BASIC PLOT TEMPLATES (WITHOUT LABELS) FOR VARIOUS GRAPHS

def plot_heatmap(dict_mat, ax):
    '''
    Applies the heatmap error arrays from a .mat imported dictionary to ax to create a colour map.
    '''
    
    u_mesh = dict_mat['u']
    v_mesh = dict_mat['v']
    err_mesh = dict_mat['err']
    
    # Process z_array by scaling with logorithms. Note norm_array > 0.
    power = 0.5
    z_mesh = (np.log(1 + err_mesh))**power
    
    
    # The bounds for z_array:
    abs_max = np.abs(z_mesh).max() #int(abs_max + 1)
    # z_min, z_max = 0, int(abs_max + 1)
    z_min, z_max = 0, abs_max
    
    c = ax.pcolormesh(u_mesh, v_mesh, z_mesh, cmap='binary', vmin=z_min, vmax=z_max) # cmap='RdBu'
    
    # Set the limits of the plot to the limits of the data
    u_min = u_mesh.min()
    u_max = u_mesh.max()
    v_min = v_mesh.min()
    v_max = v_mesh.max()
    ax.axis([u_min, u_max, v_min, v_max])
    
    # x,y-axis
    line_opt = {'linestyle': 'dashed',
                'linewidth': 0.8,
                'color': 'white'
                }

    ax.axhline(y=0, **line_opt)
    ax.axvline(x=0, **line_opt)
    
    # Plot base eigenvalue
    base_eig = dict_mat['base'][0]
    ax.plot(np.array([base_eig]),
            np.array([0]),
            marker='o',
            markersize=4,
            color='orange')
    
    # Return colour scheme
    return c 


def heatmap_positions():
    '''
    Returns a BBox with the heatmap ax positions.
    '''
    
    bbox1 = np.array([[0.05, 0.05], [0.65, 0.45]])
    bbox2 = np.array([[0.05, 0.55], [0.65, 0.95]])
    bbox3 = np.array([[0.80, 0.05], [0.90, 0.95]])
    
    bbox_list = [tsfm.Bbox(bbox1),
                 tsfm.Bbox(bbox2),
                 tsfm.Bbox(bbox3)
                 ]
    
    return bbox_list


if __name__ == '__main__':
    pass

    