from __future__ import division, print_function

import os
import numpy as np
import math
from math import pi

from library import *

# NUMERICALLY COMPUTES THE DATA ARRAYS FOLLOWING THE DDE, READY TO BE PLOTTED.

class KuraComputer(kura_pack.DDEStruct):
    '''
    A class to generate the numerical solutions and import/export the results into appropriate directories.
    Uses kura_pack.DDEStruct as a base class.
    '''
    
    def __init__(self):
        
        # Initialize base class
        kura_pack.DDEStruct.__init__(self)
        
        # Solution
        self.sol = num_pack.DiscreteFun()
        
        # Delay
        self.delay_data = num_pack.TauData()
        
        # Computations from solution and delay
        self.delta = np.array([])
        
    
    def compute_Omega(self, delta, store=True, steps=30):
        '''
        Computes the global frequency Omega using the theoretical and sample equation.
        Here, delta is the variance to use in the leading-order approximation.
        '''
        
        g = self.g
        w0 = self.omega0
        T = self.Lambda[1] / 2
        bara = 1 - self.inj
        gain = self.gain
        
        A = self.A
        Tau0 = self.Tau0
        
        self.Omega = fun_pack.plastic_freq(g, w0, T, bara, gain, delta, steps=steps)
        self.Omega_s = fun_pack.global_freq_sam(g, w0, A, Tau0, steps=steps)
        
        return self.Omega, self.Omega_s
    
    
    def generate_sol(self, get_delay=True):
        '''
        Using the current lambda functions stored and parameters, generates a numerical solution and
        delay arrays.
        '''
        
        # DDE set
        dde_set = num_pack.DDESet()
        
        dde_set.N = self.N
        dde_set.tspan = self.tspan
        dde_set.Tau0 = self.Tau0
        dde_set.step_size = self.step_size
        
        dde_set.A = self.A
        dde_set.A_inj = self.A_inj
        
        dde_set.dde_fun = self.dde_fun
        dde_set.dde_fun_inj = self.dde_fun_inj
        
        dde_set.Tau_fun = self.Tau_fun
        dde_set.hist_fun = self.hist_fun
        
        dde_set.is_inj = self.is_inj
        dde_set.inj_time = self.inj_time
        
        # Compute delay statistics
        dde_set.get_delay = get_delay
        
        # Tau categorize and sample
        dde_set.Tau_steps = self.Tau_steps
        dde_set.Tau_upper = self.Tau_upper
        
        # Tau sample (among pre_inj and post_inj Tau)
        dde_set.sam_num = 100
        dde_set.obtain_sample()
        
        sol, delay_data = num_pack.dde25(dde_set)
        
        self.sol = sol
        self.delay_data = delay_data
    
    
    def compute_var(self):
        '''
        Using the stored solution, computes the variance of the solution (phases) over time, assuming centering at 0.
        '''
        
        self.delta = comp_pack.sample_var_diff(self.sol.Y, mod=pi)
        
        
    def export_data(self):
        '''
        Exports options, solution, data into dir_save directory. Uses basenames dde_options, sol, Tau.
        '''
        
        # Options
        self.export_options('dde_options.txt')
        
        # Connections
        self.export_A()
        
        # Solution
        self.sol.save_opts = self.saveopts
        self.sol.basename = 'sol'
        self.sol.export_data(self.dir_save)
        
        # Delay data
        self.delay_data.save_opts = self.saveopts
        self.delay_data.basename = 'Tau'
        self.delay_data.export_data(self.dir_save)
    
        # Delta (Variance)
        np.savetxt(self.dir_save + '\\var.gz', self.delta, **self.saveopts)

if __name__ == '__main__':
    kc = KuraComputer()