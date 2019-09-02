from __future__ import division, print_function

import os
import numpy as np
import math
from math import pi

# PROVIDES FUNCTIONS TO PROCESS DATA

def order(Y):
    '''
    Returns the order parameter along each row of Y, assuming Y is an M by N matrix.
    '''
    
    expY = np.exp(1j*Y)
    
    N = expY.shape[1]
    orderY = np.sum(expY, 1) / N
    return orderY


def sample_mean_unmod(Y):
    '''
    Computes the sample mean of Y across each row. Mods each entry so that they are all in [-mod, mod].
    '''
    
    N = Y.shape[1]
    meanY = np.sum(Y, 1) / N    
    
    return meanY


def sample_mean(Y, mod=pi):
    '''
    Computes the sample mean of Y across each row. Mods each entry so that they are all in [-mod, mod].
    '''
    
    # Compute raw sample mean
    N = Y.shape[1]
    meanY = np.sum(Y, 1) / N
    
    if type(mod) == float:
        mean_modY = mod_bound(meanY, mod)
    else:
        mean_modY = meanY
        
    return mean_modY


def sample_var(Y):
    '''
    Computes the sample variance of Y across each row. Mods all differences so that they are all
    in [-mod, mod].
    '''
    
    N = Y.shape[1]
    meanY = sample_mean(Y)
    varY = np.sum(Y**2, 1) - N*sample_mean(Y)**2
    varY = varY / (N-1)
    return varY

    
def sample_var_diff(Y, mod=pi):
    '''
    Given Y is an M by N matrix, returns a 1-D array of length M that has the sample variance over
    each row.
    '''
    
    M, N = Y.shape
    Y_var = np.zeros(M)
    
    for j in range(M):
        
        # Row matrix
        Y_j = Y[j]
        
        # Obtain difference array
        diff_Y_j = diff_array(Y_j, mod=mod)
        
        # Sum the squares, then normalize:
        NN = diff_Y_j.size
        Y_var[j] = np.sum(diff_Y_j**2) / NN
        
    return Y_var


def sample_var_all(Y):
    '''
    Computes the single sample variance of Y over the entire array.
    '''
    
    NN = Y.size
    meanY = np.sum(Y) / NN
    varY = np.sum(Y**2) - NN*meanY**2
    varY = varY / (NN - 1)
    
    return varY


def asy_value(Y, asy=0.1):
    '''
    Given a 1-D array Y, computes the asymptotic value (the last asy indices of Y.
    Returns the asymptotic mean and asymptotic sample variance.
    '''
    
    N = Y.size
    ind = int(N*(1 - asy))
    
    Y_asy = Y[ind:]
    N_asy = Y_asy.size
    
    mean_Y_asy = np.sum(Y_asy) / N_asy
    var_Y_asy = np.sum((Y_asy - mean_Y_asy)**2) / (N_asy - 1)
    
    return mean_Y_asy, var_Y_asy


def asy_range(Y, asy=0.1):
    '''
    Given a 1-D array Y, returns the min and max of the last asy indices of Y.
    '''
    
    N = Y.size
    ind = int(N*(1 - asy))
    Y_asy = Y[ind:]
    
    return np.array([np.min(Y_asy), np.max(Y_asy)])


def diff_array(Y, mod=pi):
    '''
    Obtains the difference array between all entries of 1-D array Y.
    '''
    
    diffY = Y - np.array([Y]).T
    
    if type(mod) == float:
        mod_diffY = mod_bound(diffY, mod)
    else:
        mod_diffY = diffY
        
    return mod_diffY


def partition_array(C, K, M, conn=False):
    '''
    Categorizes the array C into M evenly spaced right-closed intervals of step-size K/M
    from 0 to K, concatenated with all X = 0 as the first entry. Returns the frequency array. 
    If conn is a matrix of ones and zeros, only considers the entries where conn = 1.
    '''
    
    # Connection matrix
    if type(conn) != bool:
        A = conn
        
    else:
        A = np.ones(size=C.size)
        
    # Step-size
    h = K / M
    
    C_int = np.int64(np.ceil(C / h))
    freq_array = np.zeros(M+1)
    
    for j in range(M):
        freq_array[j+1] = np.count_nonzero((C_int == (j+1))*(A == 1))
    
    freq_array[0] = np.count_nonzero((C_int <= 0)*(A == 1))
    
    return freq_array


def categorize_array(X, part_array):
    '''
    Categorizes the array X into partitions given by part_array. Assumes that every entry in X
    is in some interval of part_array. Returns the counts corresponding to the right-closed intervals 
    of the partition.
    '''
    
    N = part_array.size
    count_array = np.zeros(N-1)
    
    # Categorize:
    for j in range(N-1):
        X_low = np.int64(X > part_array[j])
        X_high = np.int64(X <= part_array[j+1])
        X_count = np.sum(X_low*X_high)
        count_array[j] = X_count
    
    return count_array


def midpoints(X):
    '''
    Given a 1-D array of partitions, returns an array of all midpoints of X, of size X.size - 1.
    '''
    
    X_up = X[1:]
    X_low = X[:-1]
    
    X_mid = X_low + (X_up - X_low)/2
    
    return X_mid


def concat_array(X, cols):
    '''
    Given a 1-D array X, writes the X array along each column in an N by M matrix, where
    N = X.size, M = cols.
    '''
    
    N = X.size
    ind_N = np.int64(np.arange(0,N))
    Od_N = np.int64(np.zeros((N,N)))
    
    # Square matrix
    mat_X = np.array([X])
    mat_X = mat_X[Od_N, ind_N]
    
    mat_X = mat_X.T
    
    return mat_X[:, :cols]


# SUPPLEMENTARY FUNCTIONS

def mod_array(Y, mod):
    '''
    Mods each entry of the array Y with mod.
    '''
    
    h = mod
    Y_int = np.sign(Y)*np.floor(np.abs(Y)/h)
    Y_rem = Y - Y_int*h
    
    return Y_rem


def mod_bound(Y, mod):
    '''
    Mods Y by taking the decomposition Y = k*mod + Y_rem, where k is some integer and
    Y_rem is the modded form of Y bounded by [-mod, mod]. 
    '''
    
    # Try modding by 2*mod and use if statement
    
    # Get remainder over 2*mod:
    Y_rem = mod_array(Y, 2*mod)
    
    # Convert [mod, 2*mod] to [-mod,0]:
    Y_pos = np.int64(Y_rem < mod)
    Y_neg = np.int64(Y_rem >= mod)
    
    Y_out = Y_pos*Y_rem + Y_neg*(Y_rem - 2*mod)
    
    return Y_out


if __name__ == '__main__':
    num = 10
    
    a = np.random.uniform(low=10, high=20, size=num)
    b = sample_var_mod(a)
    
