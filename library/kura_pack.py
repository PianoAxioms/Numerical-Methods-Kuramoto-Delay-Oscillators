from __future__ import division, print_function

import os
import numpy as np
from scipy import optimize
import math
from math import pi

# PROVIDES THE FRAMEWORK CLASSES THAT STORE PARAMETERS, MANAGES DATA FILES WITH IMPORT/EXPORT METHODS

class DDEStruct():
    '''
    Provides the elements to implement a dde24 numerical solver. That is, all relevant parameters and dde_fun, history.
    '''
    
    def __init__(self):
        
        # Parameters (with default values):
        N = 10
        self.N = 10
        self.g = 9.5
        self.tspan = [0,2]
        
        # Coefficients
        self.A = np.ones((N,N))
        self.A_sym = False
        self.inj = 0.0
        
        # Midway injury
        self.A_inj = np.ones((N,N))
        self.is_inj = False
        self.inj_time = 2.0
        self.inj_mid = 0.5
        
        # Omega
        self.omega0 = 20.0
        self.omega = self.omega0*np.ones(N)
        self.xi = self.omega / (2*pi)
        
        # Tau
        self.Tau0 = np.zeros((N,N))
        self.Lambda = [0.0, 1.0] # Lambda = Tau distribution parameters
        self.Tau_dist = 'uniform'
        self.Tau_sym = False
        
        # Tau parameters
        self.alpha_Tau = 0
        self.gain = 0.01
        
        # Tau distribution parameters
        self.Tau_steps = 100
        self.Tau_upper = 10.0
        
        # Function for dde_24, dde25 (in array form)
        self.dde_fun = lambda t, Y, Ytau: 0
        self.dde_fun_inj = lambda t, Y, Ytau: 0
        
        # Tau ODE function for dde25
        self.Tau_fun = lambda t, Tau, Y: np.zeros((N,N))
        
        # Extra function
        self.order_fun = lambda Y: np.sum(np.exp(1j*Y)) / Y.size
        
        # History function
        self.hist_fun = lambda t: np.zeros((N,N))
        self.hist_type = 'half_unity_linear'
        
        # Step size
        self.step_size = 0.01
        
        # Computed attributes
        self.Omega = 0.0
        self.Omega_s = 0.0
        
        self.Taum = np.zeros((N,N))
        self.Tauf = np.zeros((N,N))
        self.Tau_dt = np.array([])
        
        # Save options
        self.savefmt = '.gz'
        self.saveopts = {'fmt': '%.8e'} # 6 decimal places
        
        # Base directory
        self.dir_save = ''
        
        
    def update(self, sample_A=True, sample_Tau=True):
        '''
        Using the current loaded parameters, updates the Tau sampled matrix and the ddefun.
        '''
        
        N = self.N
        self.omega = self.omega0*np.ones(N)
        self.xi = self.omega / (2*pi)        
        
        # Coeff resample:
        if sample_A:
            self.sample_A()
            
        # Tau resample:
        if sample_Tau:
            self.sample_delay()
        
        # History function:
        self.refresh_hist_fun()
        
        # DDE functions:
        self.refresh_fun()
        
        
    def sample_delay(self):
        '''
        Given the current options, samples an i.i.d. delay matrix from a distribution.
        '''
        
        N = self.N
        
        # Resample
        if self.Tau_dist == 'uniform':
            low = self.Lambda[0]
            high = self.Lambda[1]
            self.Tau0 = np.random.uniform(low=low, high=high, size=(N,N))
        
        elif self.Tau_dist == 'constant':
            self.Tau0 = self.Lambda*np.ones((N,N))
        
        else:
            self.Tau0 = np.zeros((N,N))
            
        # Make symmetric
        if self.Tau_sym:
            self.Tau0 = make_symmetric(self.Tau0)
        
    
    def sample_A(self):
        '''
        Given the current attributes, samples the coefficient matrix A from a distribution (uniform).
        '''
        
        N = self.N
        Ones_N = np.ones((N,N))
        unif_N = np.random.uniform(low=0, high=1, size=(N,N))
        
        gamma = self.inj
        gamma_mid = self.inj_mid
        mult_mat = np.int64(unif_N >= gamma)
        mult_mat_mid = np.int64(unif_N >= gamma_mid)
        
        if self.A_sym:
            mult_mat = make_symmetric(mult_mat)
            mult_mat_mid = make_symmetric(mult_mat_mid)
        
        self.A = mult_mat*Ones_N
        self.A_inj = mult_mat_mid*Ones_N
        
    
    def refresh_hist_fun(self):
        '''
        Given the current attributes, re-defines the lambda function hist_fun.
        '''
        
        # Relevant arrays for lambda functions
        N = self.N
        One_N = np.ones((N,N))
        half_unity_mat = pi*np.arange(0,N) / N
        half_unity_mat = concat_copies(half_unity_mat)
        
        # Parameters
        w0 = self.omega0
        
        if self.hist_type == 'half_unity_constant':
            self.hist_fun = lambda t: half_unity_mat
        
        elif self.hist_type == 'half_unity_linear':
            self.hist_fun = lambda t: w0*t*One_N + half_unity_mat

        elif self.hist_type == 'rand_linear':
            init_w0 = np.random.uniform(low=0, high=2*pi, size=N)
            rand_unity_mat = np.array([init_w0])
            rand_unity_mat = rand_unity_mat[Od_N, ind_N]
            
            self.hist_fun = lambda t: w0*t*One_N + rand_unity_mat
        
        else:
            self.hist_fun = lambda t: np.zeros((N,N))
               
            
    def refresh_fun(self):
        '''
        Given the current attributes, re-defines the lambda functions dde_fun and Tau_fun.
        '''
        
        N = self.N
        
        # dde_fun:
        w0 = self.omega0
        g = self.g
        A = self.A
        A_inj = self.A_inj
        Od = np.ones(N)
        self.dde_fun = lambda t, Y, Ytau: w0*Od + (g/N)*np.sum(A*np.sin(Ytau.T - Y).T, 1)
        self.dde_fun_inj = lambda t, Y, Ytau: w0*Od + (g/N)*np.sum(A_inj*np.sin(Ytau.T - Y).T, 1)
        
        # Tau_fun:
        al = self.alpha_Tau
        gain = self.gain
        Tau0 = self.Tau0
        Zd = np.zeros((N,N))
        self.Tau_fun = lambda t, Tau, Y: al*np.int64(Tau > 0)*(-(Tau - Tau0) + gain*np.sin(Y - np.array([Y]).T))
    
    
    def import_options(self, filename):
        '''
        Given a directory to a .txt file, imports parameters to use. Uses dir_save as the base
        directory path to filename.
        '''
        
        dir_file = os.path.join(self.dir_save, filename)
        lines_list = open(dir_file, 'r').read().split('\n')
        for line in lines_list:
            if len(line) == 0:
                continue
            
            new_line = read_line(line)
            attr = new_line[0]
            value = new_line[1]
            
            # Continue if attr does not exist
            if not hasattr(self, attr):
                continue
            
            # Process value to match current attr type
            attr_type = type(getattr(self, attr))
            if attr_type == list:
                attr_type = type(getattr(self, attr)[0])
            
            value = convert_string(value, attr_type)
            
            setattr(self, attr, value)
            
    
    def export_options(self, filename):
        '''
        Given a directory to a .txt file, exports parameters to use.
        '''
        
        dir_file = os.path.join(self.dir_save, filename)
        
        raw_file = open(dir_file, 'w+')
        
        # Export template
        attr_list = DDE_option_template()
              
        with raw_file as f:
            for i in range(len(attr_list)):
                attr = attr_list[i]
                if hasattr(self, attr):
                    value = getattr(self, attr_list[i])
                    f.write(attr + ' = ' + convert_to_str(value) + '\n')
                
                elif attr == 'BREAK':
                    f.write('\n')

    
    def import_options_from_dict(dict_options):
        '''
        Given a dictionary of options, loads DDEStruct with the options.
        '''
        
        for key in dict_options.keys():
            if hasattr(self, key):
                setattr(self, key, dict_options[key])
                
    
    def import_A(self):
        '''
        Imports A, A_inj from the stored directory dir_save.
        '''
        
        file_A = os.path.join(self.dir_save, 'A' + self.savefmt)
        file_A_inj = os.path.join(self.dir_save,'A_inj' + self.savefmt)
        
        self.A = np.loadtxt(file_A)
        self.A_inj = np.loadtxt(file_A_inj)
        
    
    def export_A(self):
        '''
        Exports A, A_inj from the stored directory dir_save.
        '''
        
        file_A = os.path.join(self.dir_save, 'A' + self.savefmt)
        file_A_inj = os.path.join(self.dir_save,'A_inj' + self.savefmt)
        
        np.savetxt(file_A, self.A, fmt='%i')
        np.savetxt(file_A_inj, self.A_inj, fmt='%i')
        
        
    def import_files(self, dict_files):
        '''
        Given a dictionary of attr: filename, imports the corresponding files from dir_save\filename
        into self.attr.
        '''
        
        attr_list = dict_files.keys()
        all_files = os.listdir(self.dir_save)
        
        for attr in attr_list:
            if hasattr(self, attr):
                filename = os.path.join(self.dir_save, dict_files[attr])
                if filename in all_files:
                    im_array = np.loadtxt(filename)
                    setattr(self, attr, im_array)
                    
            else:
                continue
            
        
    def export_files(self, dict_files):
        '''
        Given a dictionary of attr: filename, imports the corresponding files into dir_save\filename
        using array self.attr.
        '''
        
        attr_list = dict_files.keys()
        
        for attr in attr_list:
            if hasattr(self, attr):
                filename = os.path.join(self.dir_save, dict_files[attr])
                ex_array = getattr(self, attr)
                np.savetxt(filename, ex_array, **self.saveopts)
        
        
# EXPORT TEMPLATES

def DDE_option_template():
    '''
    Returns a list order of attributes for dde_options.txt.
    Here, 'BREAK' denotes a line break.
    '''
    
    attr_list = ['N',
                 'omega0',
                 'g',
                 'tspan',
                 'step_size',
                 'hist_type',
                 'BREAK',
                 'A_sym',
                 'inj',
                 'BREAK',
                 'is_inj',
                 'inj_time',
                 'inj_mid',
                 'BREAK',
                 'Tau_dist',
                 'Lambda',
                 'Tau_sym',
                 'alpha_tau',
                 'gain',
                 'BREAK',
                 'Omega',
                 'Omega_s',
                 'BREAK',
                 'Tau_steps',
                 'Tau_upper'
                 ]
    
    return attr_list
    
    
# SUPPLEMENTARY FUNCTIONS

def read_line(string):
    '''
    Given string is of the form X = Y, returns X,Y.
    '''
    if '=' not in string:
        return ['None_attr', '']
    
    line_list = string.split('=')
    line_list2 = []
    
    for i in range(len(line_list)):
        line_list2.append(line_list[i].strip())
    
    return line_list2


def convert_string(string, x_type):
    '''
    Given string value(s) separated by commas and type, returns the value converted.
    '''
    
    value_list = string.split(',')
    new_value_list = []
    for i in range(len(value_list)):
        
        new_value = value_list[i]
        new_value = new_value.strip()
        
        if x_type == str:
            new_value = str(new_value)
        
        elif x_type == int:
            new_value = int(new_value)
            
        elif x_type == float:
            new_value = float(new_value)
        
        elif x_type == bool:
            if string == 'False':
                new_value = False
            else:
                new_value = True
        
        new_value_list.append(new_value)
    
    # Remove from list if there is only one element:
    if len(new_value_list) == 1:
        new_value_list = new_value_list[0]
    
    return new_value_list


def convert_to_str(value):
    '''
    Given a value, returns a string version of the value. If type(value) == list, returns
    all elements separated by a comma.
    '''
    
    if type(value) == list:
        string = ''
        for i in range(len(value)):
            string += str(value[i]) + ', '
        string = string[:-2]
    
    else:
        string = str(value)
    
    return string


def make_symmetric(M, keep='top'):
    '''
    Given a square matrix M, returns a symmetric version of M by deleting the bottom half (if keep = 'top') 
    or top half (if keep = 'bot') and replacing it with the upper entries symmetrically.
    '''
    
    N = M.shape[0]
    ind_N = np.int64(np.arange(0,N))
    ind_mat = concat_copies(ind_N)
    mult_mat = np.int64(ind_mat < ind_mat.T)
    mult_mat_diag = np.int64(ind_mat <= ind_mat.T)
    
    if keep == 'top':
        mult_mat = mult_mat.T
        mult_mat_diag = mult_mat_diag.T
        
    ind_M = mult_mat*M
    ind_diag_M = mult_mat_diag*M
    
    sym_M = ind_diag_M + ind_M.T
    
    return sym_M


def concat_copies(X):
    '''
    Given a 1D-array X, returns a square matrix with X duplicated along the rows.
    '''
    
    N = X.size
    ind_N = np.int64(np.arange(0,N))
    Od_N = np.int64(np.zeros((N,N)))
    
    mat_X = np.array([X])
    mat_X = mat_X[Od_N, ind_N]
    
    return mat_X


if __name__ == '__main__':
    dde_struct = DDEStruct()
    dde_struct.alpha_Tau = 1.0
    dde_struct.gain = 200
    dde_struct.refresh_fun()
    s = np.random.uniform(low=-0.2, high=0.2, size=(10,10))
    t = np.random.normal(0, 0.1, size=10)
    u = dde_struct.Tau_fun(1, s, t)

    