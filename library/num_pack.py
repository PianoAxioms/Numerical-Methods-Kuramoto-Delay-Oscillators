from __future__ import division, print_function

import os
import numpy as np
from scipy import optimize
import math
from math import pi
from tqdm import tqdm
import time

# SCRIPTS AND BASE CLASSES FOR ALL DISCRETE NUMERICAL COMPUTATION. INCLUDES THE MAIN NUMERICAL ALGORITHM.

class DiscreteFun():
    '''
    A discrete function to store x, y values.
    '''
    
    def __init__(self):
        
        self.t = np.array([], dtype='float64')
        self.Y = np.array([], dtype='float64')
        self.YP = np.array([], dtype='float64')
        
        # Cutoff status
        self.status = 'Empty function'
        self.cutoff = None
        
        # Save options
        self.basename = 'sol'
        self.save_opts = {'fmt': '%.8e'}
        self.save_fmt = '.gz'
        
        # Computation time
        self.comp_time = {}
        
    
    def import_data(self, filepath):
        '''
        Loads all data from the relevant text files in filepath. Uses basename as the name for t, Y, YP in the form basename_X.txt, X = t, Y, YP.
        '''
        
        # Basename
        basename = self.basename
        
        # Check if the files exist:
        files_list = os.listdir(filepath)
        filename = ''
        import_array = {'t': np.array([]),
                        'Y': np.array([]),
                        'YP': np.array([])
                        }
        
        save_fmt = self.save_fmt
        for X in ['t', 'Y', 'YP']:
            filename = basename + '_' + X + save_fmt
            if filename in files_list:
                import_array[X] = np.loadtxt(filepath + '\\' + filename)
        
        self.t = import_array['t']
        self.Y = import_array['Y']
        self.YP = import_array['YP']
        
    
    def export_data(self, filepath):
        '''
        Saves all data from the relevant text files to filepath. Uses basename as the name for t, Y, YP.
        '''
        
        basename = self.basename
        save_fmt = self.save_fmt
        
        # Save t:
        file_t = filepath + '\\' + basename + '_t' + save_fmt
        np.savetxt(file_t, self.t, **self.save_opts)
        
        # Save Y:
        file_Y = filepath + '\\' + basename + '_Y' + save_fmt
        np.savetxt(file_Y, self.Y, **self.save_opts)
        
        # Save YP:
        file_YP = filepath + '\\' + basename + '_YP' + save_fmt
        np.savetxt(file_YP, self.YP, **self.save_opts)
        

class TauData():
    '''
    A class containing all relevant output data for delays, computed throughout the
    dde25 function.
    '''
    
    def __init__(self):
        
        # Distributions of Tau at time stamps
        self.dist_0 = np.array([0]) # initial
        self.dist_m = np.array([0]) # before injury
        self.dist_f = np.array([0]) # end time
        
        # Delay statistics
        self.t = np.array([0])
        self.abs_dt = np.array([0]) # Absolute summation of delay rates TauP (over time)
        self.samY = np.array([0]) # A matrix (over time) of a sample of Tau (following inds)
        self.meanY = np.array([0]) # A 1D matrix (over time) of the mean of Tau
        self.dist = np.array([0]) # Distribution of delays (over time)
        
        # Save list
        self.attr_list = ['dist_0', 
                          'dist_m', 
                          'dist_f',
                          'abs_dt',
                          'samY',
                          'meanY',
                          'dist'
                          ]
        
        # Save options
        self.basename = 'Tau'
        self.save_opts = {}
        self.save_fmt = '.gz' 
        
    
    def import_data(self, filepath):
        '''
        Given a file path and a basename, imports all data from the corresponding array files.
        '''
        
        d = {}
        basename = self.basename
        save_fmt = self.save_fmt
        for attr in self.attr_list:
            filename = basename + '_' + attr + save_fmt
            d[attr] = np.loadtxt(os.path.join(filepath, filename))
            setattr(self, attr, d[attr])
        
    
    def export_data(self, filepath):
        '''
        Given a file path and a basename, imports all data from the corresponding array files, using
        format savefmt and options saveopts.
        '''
        
        basename = self.basename
        save_opts = self.save_opts
        save_fmt = self.save_fmt
        for attr in self.attr_list:
            filename = basename + '_' + attr + save_fmt
            attr_array = getattr(self, attr)
            np.savetxt(os.path.join(filepath, filename), attr_array, **save_opts)
            
    
class DDESet():
    '''
    A structure to hold relevant options for dde25. Here, dde_fun is a function of (t, Y, Ytau), 
    where Ytau is the delay matrix of Y under Tau. Tau is an N by N matrix of delays, with the jth column 
    corresponding to the Ytau inputs for the jth node. history is a function NN_array -> NN_array that 
    returns the history entries, with ijth entry being u_j(t - t_ij). tspan = [t0, tf] is the initial time 
    and end time, dde_set is a DDESet instance of settings for this function. If injury is true (from dde_set),
    then uses dde_fun_inj at injury_time.Includes the dde_fun and tau_fun before and after injury, along with 
    the injury matrices.
    '''
    
    def __init__(self):
        self.datatype = 'float64'
        self.step_size = 0.001
        
        # Tolerance for blow-up
        self.Blowup = 1e+10        
        
        # Dimension
        self.N = N = 10
        
        # Other options
        self.tspan = [0, 2]
        self.Tau0 = np.zeros((N,N))
        
        # Connection matrices
        self.A = np.ones((N,N))
        self.A_inj = np.ones((N,N))
        
        # dde function
        self.dde_fun = lambda t, Y, Ytau: np.zeros(N)
        self.dde_fun_inj = lambda t, Y, Ytau: np.zeros(N)
        
        # Tau (ode) function
        self.Tau_fun = lambda t, Tau, Y: np.zeros((N,N))
        
        # History function
        self.hist_fun = lambda t: np.zeros((N,N))
        
        # Midway injury
        self.is_inj = False
        self.inj_time = 2.0
        
        # Get Tau statistics
        self.get_delay = True
        
        # Tau categorize and sample
        self.dist_steps = 20
        self.dist_upper = 2
        
        # Tau sample (among pre_inj and post_inj Tau)
        self.sam_num = 50
        self.sam_inds = [] # Two tuples (i1,i2,...), (j1,j2,...)
    
    
    def obtain_sample(self):
        '''
        Samples sample_num pairs of indices for Tau_ij as a 2-tuple list to store. Takes the sample
        among existing connections post-injury (where a_ij = 1)
        '''
        
        # List and count all non-zero entries in A_inj
        conn_inds_i, conn_inds_j = np.nonzero(self.A_inj)
        count_inds = np.count_nonzero(self.A_inj)
        
        # Randomly sample indices from existing connections
        num_inds = min(conn_inds_i.size, self.sam_num)
        ind_ord = np.arange(conn_inds_i.size)    
        sam_inds = np.random.choice(ind_ord, size=num_inds, replace=False)
        
        # Get sample from existing connections
        ind_list_i = conn_inds_i[sam_inds]
        ind_list_j = conn_inds_j[sam_inds]
        
        self.sam_inds = [ind_list_i, ind_list_j]


class DDEOut():
    '''
    A class to temporarily store various solutions for the dde25 algorithm.
    '''
    
    def __init__(self):
        
        self.N = N = 10
        self.Tau_steps = 20
        self.Tau_upper = 2
        
        self.datatype = 'float64'
        
        # Solution arrays
        self.t = np.array([])
        self.Y = np.array([])
        self.YP = np.array([])
        
        # Current connection matrix
        self.A = np.ones((N,N))
        self.A_size = N**2
        
        # Delay statistics arrays
        self.Tau_inds = [] # Two sequences (i1,i2,i3,...), (j1,j2,j3,...) giving (i,j) to take for tau_ij
        self.Tau_absP = np.array([]) # Absolute summation of delay rates TauP (over time)
        self.Tau_samY = np.array([]) # A matrix (over time) of a sample of Tau (following Tau_inds)
        self.Tau_meanY = np.array([]) # A 1D matrix (over time) of the mean of Tau
        self.Tau_dist = np.array([]) # A distribution array (over time) of Tau
        
        
    def fill_zeros_sol(self):
        '''
        Defines the dimensions of the solution matrices by filling them with zeros.
        '''
        
        N = self.N
        datatype = self.datatype
        chunk = int(min(100, math.floor((2**13)/N)))
        
        # Solutions
        self.t = np.zeros(chunk, datatype)
        self.Y = np.zeros((chunk, N), datatype)
        self.YP = np.zeros((chunk, N), datatype)
    
    
    def fill_zeros_delay(self):
        '''
        Defines the dimensions of the delay statistical matrices by filling them with zeros.
        '''
        N = self.N
        M = len(self.Tau_inds[0])
        M1 = self.Tau_steps + 1
        
        datatype = self.datatype
        chunk = int(min(100, math.floor((2**13)/N)))
        
        # Delay statistics
        self.Tau_absP = np.zeros(chunk, datatype)
        self.Tau_samY = np.zeros((chunk, M), datatype)
        self.Tau_meanY = np.zeros(chunk, datatype)
        self.Tau_dist = np.zeros((chunk, M1), datatype)
    
    
    def concatenate_sol(self):
        '''
        Concatenates the solution arrays with more zeros.
        '''
        
        N = self.N
        datatype = self.datatype
        chunk = int(min(100, math.floor((2**13)/N)))
        
        # Solutions
        self.t = np.concatenate((self.t, np.zeros(chunk, datatype)))
        self.Y = np.concatenate((self.Y, np.zeros((chunk, N), datatype)))
        self.YP = np.concatenate((self.YP, np.zeros((chunk, N), datatype)))
        
    
    def concatenate_delay(self):
        '''
        Concatenates the delay statistic arrays with more zeros.
        '''
        
        N = self.N
        M = len(self.Tau_inds[0])
        M1 = self.Tau_steps + 1
        
        datatype = self.datatype
        chunk = int(min(100, math.floor((2**13)/N)))
        
        ccate = np.concatenate
        
        # Delay statistics
        self.Tau_absP = ccate((self.Tau_absP, np.zeros(chunk, datatype)))
        self.Tau_samY = ccate((self.Tau_samY, np.zeros((chunk, M), datatype)))
        self.Tau_meanY = ccate((self.Tau_meanY, np.zeros(chunk, datatype)))
        self.Tau_dist = ccate((self.Tau_dist, np.zeros((chunk, M1), datatype)))
        
    
    def store_A(self, A_new):
        '''
        Updates the current connection matrix with A_new.
        '''
        
        self.A = A_new
        self.A_size = np.count_nonzero(A_new)
        
        
    def store_sol(self, nout, tnew, ynew, ypnew):
        '''
        Given current values of t, y, yp, Tau at step nout, stores the processed values into the respective
        solution arrays.
        '''
        
        self.t[nout] = tnew
        self.Y[nout] = ynew
        self.YP[nout] = ypnew
        
    
    def store_delay(self, nout, Tau_new, TauP_new):
        '''
        Given current values of Tau, TauP at step nout, stores the processed values into the respective
        delay statistic arrays.
        '''
        
        N = self.N
        
        # Consider only connected entries
        Tau_A = self.A*Tau_new
        TauP_A = self.A*TauP_new
        N0 = self.A_size
        
        # Processed values
        Tau_absP_new = np.sum(np.abs(TauP_A)) / N0**2 # Abs avg of TauP
        Tau_meanY_new = np.sum(Tau_A) / N0**2 # Mean of all Taus
        Tau_samY_new = Tau_new[self.Tau_inds[0], self.Tau_inds[1]] # Sample of Taus
        Tau_dist_new = categorize(Tau_new, self.Tau_steps, self.Tau_upper, conn=self.A) # Histogram of Tau
        
        # Store
        self.Tau_absP[nout] = Tau_absP_new
        self.Tau_meanY[nout] = Tau_meanY_new
        self.Tau_samY[nout] = Tau_samY_new
        self.Tau_dist[nout] = Tau_dist_new
        
    
    def get_result(self, nout, sol, tau_data):
        '''
        Stores the arrays in the instances sol = DiscreteFun, tau_data = TauData, up to time index nout.
        '''
        
        # Store solution
        sol.t = self.t[:nout+1]
        sol.Y = self.Y[:nout+1]
        sol.YP = self.YP[:nout+1]
        
        # Store delay statistics
        tau_data.t = self.t[:nout+1]
        tau_data.abs_dt = self.Tau_absP[:nout+1]
        tau_data.samY = self.Tau_samY[:nout+1]
        tau_data.meanY = self.Tau_meanY[:nout+1]
        tau_data.dist = self.Tau_dist[:nout+1]
        

def dde25(dde_set):
    '''
    Given the options in dde_set, computes and returns a DiscreteFun solution instance and a TauData 
    delay statistics instance.
    '''
    
    # OBJECT INITIALIZATION
    
    # All parameters
    datatype = dde_set.datatype
    
    t0 = dde_set.tspan[0]
    tf = dde_set.tspan[1]
    
    # Step-size
    h = dde_set.step_size
    
    # Dimension
    N = dde_set.N
    N_sam = dde_set.sam_inds[0].size
    
    # Initialize output class
    dde_out = DDEOut()
    dde_out.N = N
    dde_out.Tau_inds = dde_set.sam_inds
    dde_out.Tau_steps = dde_set.Tau_steps
    dde_out.Tau_upper = dde_set.Tau_upper
    
    # Initial concatenation of zeros
    dde_out.fill_zeros_sol()
    dde_out.fill_zeros_delay()
    
    # Initial functions
    dde_out.store_A(dde_set.A)
    dde_fun = dde_set.dde_fun
    hist_fun = dde_set.hist_fun
    Tau_fun = dde_set.Tau_fun
    
    
    # FIRST LOOP
    
    # Initial solution values
    t = t0
    y = hist_fun(t0)[0]
    yp = np.zeros(N)
    Tau0 = dde_set.Tau0
    ytau = lagvals(0, h, Tau0, hist_fun, np.array([t0]), np.array([y]), np.array([yp]))
    f = dde_fun(t, y, ytau)
    
    # Initial delay values
    zero_N = np.zeros((N,N)) # For taking non-negative Tau
    Tau = dde_set.Tau0
    Taup = Tau_fun(t0, Tau0, y)    
    
    
    # IINITIALIZE SOLUTIONS
    sol = DiscreteFun()
    Tau_data = TauData()
    Tau_data.dist_0 = dde_set.Tau0
    
    
    # STORE (INITIAL) VALUES
    nout =  0
    dde_out.store_sol(0, t0, y, f)
    if dde_set.get_delay:
        dde_out.store_delay(0, Tau, Taup)
     
    # Status
    status = 'Integration successful' 
    
    # Progress bar
    pbar = tqdm(total=100)
    up_num = 10
    tupdate = t0
    
    # MAIN LOOP
    done = False
    inj_occur = False
    while not done:
        
        tnew = t + h
        
        X = dde_out.t[:nout+1]
        Y = dde_out.Y[:nout+1]
        YP = dde_out.YP[:nout+1]
        
        # Step forward delay matrix, count
        Tau_new = Tau + h*Taup
        
        # Take maximum with 0 delay
        Tau_new = np.maximum(Tau_new, zero_N)
        
        # Delay solution matrix
        ytau = lagvals(nout+1, h, Tau_new, hist_fun, X, Y, YP)
        
        # Move using previous f array:
        ynew = y + h*f
        
        # Use injury function?
        if (tnew >= dde_set.inj_time) and not inj_occur:
            inj_occur = True
            Tau_data.dist_m = Tau.copy() # Store delay matrix
            
            if dde_set.is_inj:     
                dde_fun = dde_set.dde_fun_inj # Change function to injury fun
                dde_out.store_A(dde_set.A_inj)
        
        # Update f array:
        f = dde_fun(tnew, ynew, ytau)
        
        # Compute delay matrix slope:
        Taup = Tau_fun(tnew, Tau_new, ynew)
        
        
        # STORE
        nout += 1
        
        # If arrays are too small, concatenate
        if nout >= dde_out.t.shape[0]:
            dde_out.concatenate_sol()
            dde_out.concatenate_delay()
        
        # Input new values
        dde_out.store_sol(nout, tnew, ynew, f)
        if dde_set.get_delay:
            dde_out.store_delay(nout, Tau_new, Taup)
        
        # Are we done?
        if (tnew + h) >= tf:
            done = True
        
        # Update progress bar (pbar):
        if (tnew - tupdate) >= (tf - t0)/up_num:
            pbar.update(int(100/up_num))
            tupdate = tnew
            
        # Advance the integration one step
        t = tnew
        y = ynew
        Tau = Tau_new
    
        # Terminate if the blow-up is detected
        Ymax = np.max(Y)
        if abs(Ymax) > dde_set.Blowup:
            status = 'Blow-up detected'
            done = True

    # OUTPUT
    dde_out.get_result(nout, sol, Tau_data)
    Tau_data.dist_f = Tau.copy() 
    
    sol.status = status
    sol.cutoff = t
    
    # Close pbar:
    pbar.close()
    
    return sol, Tau_data


def lagvals(n, h, Tau, history, X, Y, YP):
    '''
    Computes and returns matrix YTau, using an array-based method of finding the values of Y at the delay times. Here,
    n is the step number, h is the step-size, Tau is the current delay matrix. history is a function NN_array -> NN_array that 
    returns the history entries. X, Y, YP are the times, solution, solution slope. Uses linear interpolation with YP (forward Euler).
    '''
    
    # Decompose Tau as multiple*h + remainder
    Tau_h, Tau_rem = decompose(h, Tau)
    
    # Initialize output matrix:
    N = Tau_h.shape[0]
    inds_N = np.int64(np.arange(0, N))
    od_N = np.int64(np.ones((N,N)))
    
    # Get initial, final times for X:
    t0 = X[0]
    tf = X[-1]
    
    tnow = t0 + n*h 
    t_tau = tnow - Tau
    tnow_h = n - Tau_h
    
    # t0 <= t_Tau <= tf indicators:
    ind_t0 = np.int64(tnow_h > 0)
    ind_n = np.int64(Tau_h == 0)
    ind_btw = ind_t0*(1 - ind_n)
    
    # Y matrix:
    Y_tau_r = ind_btw*Y[ind_btw*tnow_h, inds_N]
    Y_tau_l = ind_btw*Y[ind_btw*(tnow_h - 1), inds_N]
    X_tau_l = ind_btw*X[ind_btw*(tnow_h - 1)]
    YP_tau_r = ind_btw*YP[ind_btw*tnow_h, inds_N]
    YP_tau_l = ind_btw*YP[ind_btw*(tnow_h - 1), inds_N]
    
    Y_tau = ntrp3h(t_tau, X_tau_l, Y_tau_l, X_tau_l + h, Y_tau_r, YP_tau_l, YP_tau_r)
    
    # History matrix:
    hist_tau = history(t_tau)
    
    # Euler forward:
    Y_0 = Y[-1*od_N, inds_N] + h*YP[-1*od_N, inds_N]
    
    # Final matrix:
    Y_delay = Y_tau + (1 - ind_t0)*hist_tau + ind_n*Y_0
    
    return Y_delay


def ntrp3h(tnow, tl, Yl, tr, Yr, YPl, YPr):
    '''
    Interpolation helper function. Evaluates the Hermite cubic interpolant at time tnow.
    '''
    h = tr - tl
    s = (tnow - tl) / h
    s2 = s**2
    s3 = s**3
    
    # Slope
    m = (Yr - Yl) / h
    c = 3*m - 2*YPl - YPr
    d = YPl + YPr - 2*m
    
    Yint = Yl + (h*d*s3 + h*c*s2 + h*YPl*s)
    return Yint

    
def decompose(h, Y):
    '''
    Returns two arrays: Y_h, Y_remainder, under the decomposition Y = Y_h*h + Y_remainder
    '''
    
    Y_int = np.sign(Y)*np.floor(np.abs(Y)/h)
    Y_remainder = Y - Y_int*h
    Y_int = np.int64(Y_int)
    
    return Y_int, Y_remainder


def categorize(C, N, K, conn=False):
    '''
    Given a matrix C, number of steps N, upper bound K, categorizes C into N+1
    left-open partitions of [0,K] of step-size K/N and counts the frequency.
    Returns the frequency array. The first entry is where C <= 0. 
    If conn is a matrix of ones and zeros, only considers the entries where conn = 1.
    '''
    
    # Connection matrix
    if type(conn) != bool:
        A = conn
        
    else:
        A = np.ones(C.shape)
        
    # Step-size
    h = K / N
    
    C_int = np.int64(np.ceil(C / h))
    freq_array = np.zeros(N+1)
    
    for j in range(N):
        freq_array[j+1] = np.count_nonzero((C_int == (j+1))*(A == 1))
    
    freq_array[0] = np.count_nonzero((C_int <= 0)*(A == 1))
    
    return freq_array
    
    
def abs_avg(X):
    '''
    Given a matrix X, returns the absolute average value.
    '''
    
    NN = X.size
    return np.sum(np.abs(X)) / NN


if __name__ == '__main__':
    dde_set = DDESet()
    dde_set.obtain_sample()
    
    

