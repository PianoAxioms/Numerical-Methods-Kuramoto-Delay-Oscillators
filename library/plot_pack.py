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


def basic_template(ax, tick='small'):
    '''
    Loads the ax with basic plot options.
    '''
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
   
    if tick == 'small':
        ax.tick_params('x', **plot_options('tick', 'small'))
        ax.tick_params('y', **plot_options('tick', 'small'))
    
    else:
        ax.tick_params('x', **plot_options('tick', 'medium'))
        ax.tick_params('y', **plot_options('tick', 'medium'))



def centered_phases_template(ax):
    '''
    The template for centered phases plot.
    '''
    
    # Basic options
    basic_template(ax, tick='medium')
    
    # Axes
    ax.set_yticks(np.array([-pi/2, 0, pi/2]))
    ax.set_yticklabels((r'$-\pi/2$', r'$0$', r'$\pi/2$'))
    
    ax.set_ylim(bottom=-pi/2, top=pi/2)
    

def phase_diffs_template(ax):
    '''
    The template for the phase difference histogram plot.
    '''
    
    basic_template(ax, tick='medium')
    
    # Axes
    ax.set_xticks(np.array([-pi, -pi/2, 0, pi/2, pi]))
    ax.set_xticklabels((r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
    ax.set_xlim(left=-pi, right=pi)
    
    
    
# PLOT OPTIONS
def plot_options(attr, option, color='None'):
    '''
    Returns a dictionary of dictionary options for the following attr, with all options:
    - title: 'default', 'large', 'small', 'tiny'
    - label: 'default', 'large', 'small', 'tiny'
    - tick: 'default', 'small', 'tiny'
    - line: 'dashed', 'dotted, 'default', 'small'
    - marker: 'default', 'small', 'tiny',
            : 'bullet', 'triangle', 'square'
    
    The basic color options are: 'black', 'blue', red', 'green'
    '''
    
    d = {}
    if attr == 'title' or attr == 'label':
        
        if attr == 'title':
            d['verticalalignment'] = 'baseline'
        
        if option == 'default':
            d['fontsize'] = 12
        
        elif option == 'large': 
            d['fontsize'] = 16
        
        elif option == 'medium':
            d['fontsize'] = 14
            
        elif option == 'small':
            d['fontsize'] = 10
        
        elif option == 'tiny':
            d['fontsize'] = 8
        
    
    elif attr == 'tick':
        d['which'] = 'major'
        
        if option == 'default':
            d['labelsize'] = 10
        
        elif option == 'small':
            d['labelsize'] = 8
        
        elif option == 'tiny':
            d['labelsize'] = 6
        
        if color != 'None':
            d['labelcolor'] = color
    
    
    elif attr == 'line':
        
        if option == 'dashed':
            d['linestyle'] = 'dashed'
            d['linewidth'] = 0.8
        
        elif option == 'dotted':
            d['linestyle'] = 'dotted'
            d['linewidth'] = 0.5
            
        elif option == 'default':
            d['linestyle'] = 'solid'
            d['linewidth'] = 1.5
        
        elif option == 'medium':
            d['linestyle'] = 'solid'
            d['linewidth'] = 1.0
            
        elif option == 'small':
            d['linestyle'] = 'solid'
            d['linewidth'] = 0.5
    
    
    elif attr == 'marker':
        
        d['linestyle'] = 'None'
        
        # Break option into size_type:
        if '_' not in option:
            size = 'default'
            mark_type = option
        
        else:
            size, mark_type = option.split('_')
        
        # Options
        if size == 'big':
            d['markersize'] = 6
        
        elif size == 'default':
            d['markersize'] = 4
        
        elif size == 'small':
            d['markersize'] = 2
        
        elif size == 'tiny':
            d['markersize'] = 1
        
        if mark_type == 'bullet':
            d['marker'] = 'o'
        
        elif mark_type == 'triangle':
            d['marker'] = '^'
        
        elif mark_type == 'square':
            d['marker'] = 's'
            
         
    # Color
    if color != 'None':
        d['color'] = color
    
    return d


if __name__ == '__main__':
    pass

    