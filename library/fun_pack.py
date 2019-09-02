from __future__ import division, print_function

import os
import numpy as np
import math
from math import pi


# A library that accepts the relevant parameter inputs and returns the corresponding functions
# for the fixed-point equations, to be used with the fixed-point approximators.


# THEORETICAL FIXED-POINT EQUATION FUNCTIONS

def Omega_N(g, w0, A, Tau0):
    '''
    Returns the fixed-point function for global frequency Omega with N oscillators and connection
    matrix A and initial delay matrix Tau0.
    '''
    
    N2 = A.shape
    f = lambda u: w0 + (g/N2)*np.sum(A*np.sin(-1*u*Tau0))
    
    return f


def Omega_infty(g, w0, bara, T, gain, delta):
    '''
    Returns the N-limit fixed-point function for global frequency Omega with average connection bara
    and initial delays being uniformly distributed along [0, 2T]. The fixed-point is a leading order
    expansion with respect to phase difference delta. At delta = 0, we obtain the non-plastic equation.
    '''
    
    term1 = -1*delta / (np.sqrt(2*pi))
    
    if T == 0:
        term2 = lambda u: (1 - u*gain)*delta
        term3 = 0
        
        f = lambda u: w0 + g*bara*(term1 + term2(u) + term3)
    
    else:
        term2 = lambda u: np.sin(2*u*T)*(1-u*gain)*delta / (2*u*T*np.sqrt(2*pi))
        term3 = lambda u: (np.cos(2*u*T) -  1)/(2*u*T)*np.exp(-0.5*((1-u*gain)**2)*delta**2)
        
        f = lambda u: w0 + g*bara*(term1 + term2(u) + term3(u))
    
    return f


def eig_base(Omega, g, bara, T, plas=True):
    '''
    Returns the base eigenvalue, under a uniform initial delay distribution [0, 2T].
    '''
    
    if T != 0:
        fac = -g*bara*np.sin(2*Omega*T) / (2*T)
    else:
        fac = -g*bara
    
    if plas:
        eig = -g*bara/2 + fac/2
    else:
        eig = fac
        
    return eig


def eig_infty(g, Omega, T, bara, sigma, plas=True):
    '''
    Returns the eigenvalue function on the right-side, with respect to lambda, given that we have
    a uniform initial delay distribution.
    '''
    
    term1 = lambda u: (np.exp(-2*u*T)*(Omega*np.sin(2*Omega*T) - u*np.cos(2*Omega*T)) + u) / Omega**2
    term = lambda u: (1 + (u / Omega)**2)**(-1)*term1(u)
    
    f1 = lambda u: (g/ 2*T)*(sigma*term(u) - bara*np.sin(2*Omega*T) / Omega)
    
    if plas:
        f = lambda u: g*(sigma - bara)/2 + f1(u) / 2
    else:
        f = f1
        
    return f


# FUNCTIONS FOR NUMERICAL INTEGRATION

def kura_fun(g, w0, A):
    '''
    Given the parameters (as inputs), returns the DDE function for Kuramoto phases in the DDE.
    '''
    
    N = A.shape[0]
    Od = np.ones(N)
    
    f = lambda t, Y, Ytau: w0*Od + (g/N)*np.sum(A*np.sin(Ytau.T - Y).T, 1)
    return f


def kuraP_fun(g, w0, A):
    '''
    Given the parameters (as inputs), returns the derivative of the DDE function for Kuramoto phases in the DDE.
    To be used as the 2nd-order DDE y'' = f'(t, y, ytau, yp, yptau, tau, taup) for the acceleration.
    '''
    
    pass


def tau_fun(gain, Tau0, alpha):
    '''
    Given the parameters (as inputs), returns the ODE function for the delays Tau.
    '''

    # Tau_fun:
    N = Tau0.shape[0]
    al = alpha
    Zd = np.zeros((N,N))
    f = lambda t, Tau, Y: al*np.int64(Tau > 0)*(-(Tau - Tau0) + gain*np.sin(Y - np.array([Y]).T))
    
    return f

def hist_fun(N, w0, t0, style):
    '''
    Produces the history function given by (t, i) -> phi_i(t), i <= N, t < t0. Here, w0 is the natural frequency and style
    determines the type of history function is produced:
    - 'unif_const': phi_i(t) = pi*i / N
    - 'rand_const': phi_i(t) = phi_i, phi_i ~ Unif[0, 2*pi]
    - 'unif_linear': phi_i(t) = pi*i / N + w0*(t - t0)
    - 'rand_linear': phi_i(t) = phi_i + w0*(t - t0), phi_i ~ Unif[0, 2*pi]
    '''
    
    if style == 'unif_const':
        phi = lambda t, i: pi*i / N
    
    elif style == 'rand_const':
        rand = 2*pi*np.random.uniform(N)
        phi = lambda t, i: rand[i]
    
    elif style == 'unif_linear':
        phi = lambda t, i: pi*i / N + w0*(t - t0)
    
    elif style == 'rand_linear':
        rand = 2*pi*np.random.uniform(N)
        phi = lambda t, i: rand[i] + w0*(t - t0)
    
    else:
        phi = lambda t, i: 0
        
    return phi


# TEST
def hist_mat(N):
    '''
    Returns a unif_const history matrix, with the ij entry being phi_j(t) (only dependent on column)
    '''
    
    X = pi*np.arange(N) / N
    ind_N = np.int64(np.arange(0,N))
    Od_N = np.int64(np.zeros((N,N)))
    
    mat_X = np.array([X])
    mat_X = mat_X[Od_N, ind_N]
    
    phi = lambda t: mat_X
    
    return phi


if __name__ == '__main__':
    pass
    