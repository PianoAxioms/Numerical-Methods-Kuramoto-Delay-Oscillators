from __future__ import division, print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import math

from math import pi
from library import *

# PROVIDES ALL CLASSES AND METHODS NECESSARY FOR CREATING A SIMPLE SCRIPT TO GENERATE, IMPORT/EXPORT, AND PLOT DATA.
# FOR FIGURE 3-6

class Data(kura_pack.DDEStruct):
    '''
    A subclass to import, store, and process solution data, ready to be plotted.
    '''
    
    def __init__(self):
        
        # Initialize baseclass
        kura_pack.DDEStruct.__init__(self)
        
        # Solutions
        self.sol = num_pack.DiscreteFun()
        self.delay_data = num_pack.TauData()
        
        # Parameters
        self.num_inds = 30 # Number of phase samples
        self.diff_steps = 50 # Number of steps in phase difference plot
        self.gauss_steps = 200 # Number of steps in Gaussian function plot
        self.dist_upper = 10 # Upper bound for delay density
        self.dist_steps = 50 # Number of steps for delay density
        self.back_t = -0.15 # Backtrack time to make t = 0 more clear on delay dist
        
        # Processed data
        self.var = np.array([]) # To be imported
        
        # Mean arrays
        self.mean_phase = np.array([])
        self.mean_freq = np.array([])
        self.log_delay_dt = np.array([])
        
        # Sampled phases centered
        self.phase_inds = np.array([])
        self.centered_phases = np.array([])
        
        # Phase differences
        self.phase_diff_x = np.array([])
        self.phase_diff_0 = np.array([])
        self.phase_diff_m = np.array([])
        self.phase_diff_f = np.array([])
        self.phase_gauss_x = np.array([])
        self.phase_gauss_y = np.array([])
        
        self.phase_diff_var = 0
        
        # Delay distribution
        self.Tau_space = np.array([])
        self.Tau0_count = np.array([])
        self.Taum_count = np.array([])
        self.Tauf_count = np.array([])
        
        # Line options
        self.linecolor = 'black'
        
        
    def import_data(self):
        '''
        Imports all data from the directory dir_save.
        '''
        
        self.import_A()
        self.import_options('dde_options.txt')
        
        # Import solutions
        dir_save = self.dir_save
        self.sol.import_data(dir_save)
        self.delay_data.import_data(dir_save)
        
        self.var = np.loadtxt(self.dir_save + '\\var' + self.savefmt)
        
    
    def compute_arrays(self):
        '''
        After importing data, process all arrays to be plotted.
        '''
        
        self.compute_mean_arrays()
        self.compute_center_phase_arrays()
        self.compute_phase_diff_arrays()
        self.compute_delay_dists()
        

    def compute_mean_arrays(self):
        '''
        After importing data, computes the mean and summation arrays.
        '''
        
        # Mean phase and freq
        self.mean_phase = comp_pack.sample_mean(self.sol.Y, mod=pi)
        self.mean_freq = comp_pack.sample_mean(self.sol.YP, mod=None)
    
        # Log delay rate
        self.log_delay_dt = np.log(1 + self.N**2*self.delay_data.abs_dt)
        
        
    def compute_center_phase_arrays(self):
        '''
        After importing data, computes the centered phase arrays.
        '''
        
        # Centered sampled phases
        M = self.num_inds
        self.phase_inds = np.random.choice(np.arange(self.N), 
                                           size=self.num_inds, 
                                           replace=False)
        
        Y_ind_array = self.sol.Y[:, self.phase_inds]
        Y_mean_array = comp_pack.sample_mean_unmod(self.sol.Y)
        Y_mean_copied_array = comp_pack.concat_array(Y_mean_array, M)
        Y_center_array = Y_ind_array - Y_mean_copied_array
        Y_mod_array = comp_pack.mod_bound(Y_center_array, pi)
        self.centered_phases = Y_mod_array
    
    
    def compute_phase_diff_arrays(self):
        '''
        After importing data, computes the phase difference arrays at the end of
        simulation time.
        '''
        
        # Obtain mid time index
        mid_ind = np.argmin(np.abs(self.sol.t - self.inj_time))
        
        # Phase differences
        phase0_array = self.sol.Y[0]
        phasem_array = self.sol.Y[mid_ind]
        phasef_array = self.sol.Y[-1]
        
        phase0_diffs = comp_pack.diff_array(phase0_array, mod=pi)
        phasem_diffs = comp_pack.diff_array(phasem_array, mod=pi)
        phasef_diffs = comp_pack.diff_array(phasef_array, mod=pi)
        
        # Obtain sample variance at final time
        self.phase_diff_var = sig2 = comp_pack.sample_var_all(phasef_diffs)
        
        # Obtain theoretical Gaussian line
        self.phase_gauss_x = gauss_x = np.linspace(-pi, pi, num=self.gauss_steps)
        self.phase_gauss_y = (1/np.sqrt(2*pi*sig2))*np.exp(-gauss_x**2/(2*sig2))
        
        # Obtain density (histogram) of phase differences.
        diff_x_array = np.linspace(-pi, pi, num=self.diff_steps)
        self.phase_diff_0 = comp_pack.categorize_array(phase0_diffs, diff_x_array)
        self.phase_diff_m = comp_pack.categorize_array(phasem_diffs, diff_x_array)
        self.phase_diff_f = comp_pack.categorize_array(phasef_diffs, diff_x_array)
        
        # Adjust phase difference of x
        self.phase_diff_x = comp_pack.midpoints(diff_x_array)
        
        # Normalize
        norm_0 = phase0_diffs.size*2*pi / self.diff_steps
        self.phase_diff_0 = self.phase_diff_0 / norm_0
        
        norm_m = phasem_diffs.size*2*pi / self.diff_steps
        self.phase_diff_m = self.phase_diff_m / norm_m
        
        norm_f = phasef_diffs.size*2*pi / self.diff_steps
        self.phase_diff_f = self.phase_diff_f / norm_f
        
    
    def compute_delay_dists(self):
        '''
        After importing data, computes the delay distributions at initial,
        middle, and final times, along with a concatenated delay array to
        produce a coloured histogram.
        '''
        
        K = self.dist_upper
        M = self.dist_steps
        
        # Obtain x array
        x_array = np.linspace(0, K, num=M+1)
        
        # Obtain count arrays
        freq = comp_pack.partition_array
        Tau0_count = freq(self.delay_data.dist_0, K, M, conn=self.A)
        Taum_count = freq(self.delay_data.dist_m, K, M, conn=self.A)
        Tauf_count = freq(self.delay_data.dist_f, K, M, conn=self.A_inj)
        
        # Concatenate
        concat = np.concatenate
        back_t = self.back_t
        arr0 = np.array([0,0])
        self.Tau_space = concat((x_array, np.array([x_array[-1], back_t, back_t])))
        self.Tau0_count = concat((Tau0_count, arr0, np.array([Tau0_count[1]])))
        self.Taum_count = concat((Taum_count, arr0, np.array([Taum_count[0]])))
        self.Tauf_count = concat((Tauf_count, arr0, np.array([Tauf_count[0]])))
        
    
    def get_color_ratio(self, attr, upper):
        '''
        Returns a float between 0.0 and 1.0 based on self.attr / upper.
        '''
        
        value = getattr(self, attr)
        return value / upper


class SubData():
    '''
    The data class, but only includes attributes relevant to what will be plotted.
    '''
    
    def __init__(self):
        
        self.t = np.array([])
        self.net_tau = np.array([])
        
        self.meanYP = np.array([])
        self.delta = np.array([])
        
        self.gain = 0.0
        self.inj_mid = 0.0
        
        # Asymptotic values
        self.asy_Omega_mean = 0.0
        self.asy_Omega_var = 0.0
        self.asy_Omega_range = [0, 1]
    
        self.asy_delta_mean = 0.0
        self.asy_delta_var = 0.0
        self.asy_delta_range = [0, 1]
        
        
    def process_data(self, data_inst):
        '''
        Given an instance of the Data class, takes the relevant attributes and stores it in the class.
        Make sure to apply the compute_mean_arrays method on the Data class first.
        '''
        
        self.t = data_inst.sol.t
        self.net_tau = data_inst.log_delay_dt
        
        self.meanYP = data_inst.mean_freq
        self.delta = np.sqrt(data_inst.var)
        
        self.gain = data_inst.gain
        self.inj_mid = data_inst.inj_mid
        
    
    def compute_asy(self, asy=0.1):
        '''
        Using the stored meanYP and delta arrays, stores the asymptotic mean of meanYP and delta, along with their
        min/max range, over the last asy*100% of their indices.
        '''
        
        # YP
        self.asy_Omega_mean, self.asy_YP_var = comp_pack.asy_value(self.meanYP, asy=asy)
        self.asy_Omega_range = comp_pack.asy_range(self.meanYP, asy=asy)
        
        # Delta
        self.asy_delta_mean, self.asy_delta_var = comp_pack.asy_value(self.delta, asy=asy)
        self.asy_delta_range = comp_pack.asy_range(self.delta, asy=asy)
        
    
class Processor():
    '''
    A class containing methods to compile multiple data sets and store into the appropriate arrays for graphing.
    To be used for multiple simulation plots.
    '''
    
    def __init__(self):
        
        # An instance of the Data class for importing
        self.dm = Data()
        
        # A dictionary of sub-class instances to compile the plot arrays from
        self.data = {}
        
        # Asymptotic arrays
        self.gain_array = np.array([])
        self.inj_array = np.array([])
        
        self.asy_Omega = np.array([])
        self.asy_Omega_range = np.array([])
        
        self.asy_delta = np.array([])
        self.asy_delta_range = np.array([])
        
        # Parameters
        self.omega0 = 0.0
        self.inj_time = 1.0
        
        # Data set directory
        self.dir_save = ''
    
    
    def save_data(self, key, save_param=False):
        '''
        Using the current stored data, store all relevant data into the appropriate arrays.
        '''
        
        # Process
        self.dm.compute_mean_arrays()
        
        # Create new Subdata instance
        self.data[key] = SubData()
        
        # Store
        self.data[key].process_data(self.dm)
        
        if save_param:
            self.omega0 = self.dm.omega0
            self.inj_time = self.dm.inj_time
    
    
    def save_dataset(self):
        '''
        Using the current data set directory, imports from each trial folder and saves all relevant data.
        '''
        
        # Get directory list
        all_files = os.listdir(self.dir_save)
        data_files = []
        
        # Filter files
        for x in all_files:
            y = x.split('.')
            if len(y) == 1:
                data_files.append(x)
        
        # Import from each folder
        save_param = True
        for x in data_files:
            x_dir = os.path.join(self.dir_save, x)
            self.dm.dir_save = x_dir
            
            self.dm.import_data()
            
            # Check if folder is an integer
            if x.isnumeric():
                key = int(x)
                self.save_data(key, save_param=save_param)
                save_param = False
    
    
    def compile_asy_arrays(self, asy=0.1):
        '''
        Compiles and returns the asymptotic (last asy indices) arrays, along with the gain and injury arrays.
        '''
        
        # Sort the keys in increasing order
        keys = list(self.data.keys())
        keys.sort()
        
        # Define return arrays
        self.gain_array = np.zeros(len(keys), dtype='float64')
        self.inj_array = np.zeros(len(keys), dtype='float64')
        
        self.asy_Omega = np.zeros(len(keys), dtype='float64')
        self.asy_Omega_range = np.zeros((2, len(keys)), dtype='float64') # Use X[k]
        
        self.asy_delta = np.zeros(len(keys), dtype='float64')
        self.asy_delta_range = np.zeros((2, len(keys)), dtype='float64')
        
        # Extract data points
        for i in range(len(keys)):
            key = keys[i]
            
            # Compute and store asymptotic values
            self.data[key].compute_asy(asy=asy)
            
            # Get x-arrays
            self.gain_array[i] = self.data[key].gain
            self.inj_array[i] = self.data[key].inj_mid
            
            # Get the data point
            self.asy_Omega[i] = self.data[key].asy_Omega_mean
            self.asy_Omega_range[:,i] = self.data[key].asy_Omega_range
            
            self.asy_delta[i] = self.data[key].asy_delta_mean
            self.asy_delta_range[:,i] = self.data[key].asy_delta_range
            
            
if __name__ == '__main__':
    dm = Data()
