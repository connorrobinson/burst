import numpy as np
import matplotlib.pyplot as plt

'''

fluid.py

Includes useful equations for working with the coordiante system used in the simulations of Robinson 2017. 

AUTHOR:
    Connor Robinson, August 22nd, 2019

'''

def get_theta(r, rdisk, GAMMA, Z = 0.05):
    theta_o = np.pi/2 - np.arctan(Z/rdisk)
    p,q = get_pq(rdisk, theta_o, GAMMA)
    theta = np.arcsin(np.sqrt( 2.0/(5.0*GAMMA) * ((r**2.0 + GAMMA) - ((r**2.0 + GAMMA)**2.0 - 5.0*GAMMA*q*r**3.0)**(1.0/2.0))))
    return theta

def get_pq(r,theta, GAMMA):
    p = (-1.0/4.0) * (r**(-4.0)) * GAMMA * (5.0*np.cos(theta)**(2.0) - 3.0) * np.cos(theta) - (r**(-2.0)) * np.cos(theta)
    q = (1.0/4.0) * (r**(-3.0)) * GAMMA * (5.0*np.cos(theta)**(2.0) - 1.0) * np.sin(theta)**(2.0) + (r**(-1.0)) * np.sin(theta)**(2.0)
    return p, q

def get_x(rdisk, GAMMA, Z = 0.05, npoint = 500):
    r = np.linspace(1,rdisk, npoint)
    theta = get_theta(r, rdisk, GAMMA, Z)
    return r * np.sin(theta)

def get_y(rdisk, GAMMA, Z = 0.05, npoint = 500):
    r = np.linspace(1,rdisk, npoint)
    theta = get_theta(r, rdisk, GAMMA, Z)
    return r * np.cos(theta)

def get_scale(r, theta, GAMMA):
    '''
    Calculate the scale factors hp, hq and h3 for octupole + dipole 
    
    '''
    
    f = get_f(r, theta, GAMMA)
    g = get_g(r, theta, GAMMA)
    
    hp = r**5 * (f**2*np.cos(theta)**2 + g**2*np.sin(theta)**2)**(-1/2)
    h3 = r*np.sin(theta)
    hq = r**4/np.sin(theta)*(g**2*np.sin(theta)**2 + f**2.*np.cos(theta)**2)**(-1/2)
    
    return hp, hq, h3
    
def get_f(r, theta, GAMMA):
    '''
    Calculate the ancillary function f for octopole + dipole coordinates
    
    '''
    f = GAMMA * (5*np.cos(theta)**2 - 3) + 2*(r**2)
    return f

def get_g(r, theta, GAMMA):
    '''
    Calculate the ancillary function g for octopole + dipole coordinates
    
    '''
    g = (3/4) * GAMMA * (5*np.cos(theta)**2 - 1) + r**2
    return g

def get_omega(b, rdisk):
    '''
    Calculates the dimensionless rotation parameter
    
    '''
    omega = b/rdisk**3
    
    return omega

def get_xdotp(r,GAMMA,theta):
    '''
    Calculates x . p for dipole + octopole geometry 
    
    '''
    f = get_f(r, theta, GAMMA)
    g = get_g(r, theta, GAMMA)
    
    H = get_H(r,theta,GAMMA)
    
    xdotp = (r**(-5) * np.cos(theta) * np.sin(theta) * (f + g)) / np.sqrt(H)
    
    return xdotp

def get_H(r,theta,GAMMA):
    '''
    Calculate the ancillary function H for octopole + dipole coordinates
    
    '''
    f = get_f(r, theta, GAMMA)
    g = get_g(r, theta, GAMMA)
    
    H = r**(-10)*(np.cos(theta)**2 * f**2 + np.sin(theta)**2 * g**2)
    
    return H

def get_gravity(r,theta,GAMMA,b):
    '''
    Calculates the gravitional acceleration along the column.
    
    '''
    
    hp, hq, h3 = get_scale(r, theta, GAMMA)
    
    G = 1
    
    gr = -G * b / r**2
    
    H = get_H(r,theta,GAMMA)
    f = get_f(r,theta,GAMMA)
    
    dr_dp = r**(-5) * f * np.cos(theta) / H
    
    g = gr * dr_dp / hp
    
    return g

def get_rot(r, GAMMA, theta, b, rdisk):
    '''
    Calculates the centrifugal acceleration along the column
    
    '''
    
    xdotp = get_xdotp(r,GAMMA,theta)
    
    omega = get_omega(b, rdisk)
    
    rot = omega * r * np.sin(theta) * xdotp
    
    return rot

def get_b(M, R, T):
    '''
    Get the depth of the potential well. Takes values in solar units
    
    '''
    
    G = 6.67e-8
    
    Msol = 2e33
    Rsol = 6.96e10
    
    Mstar = M * Msol
    Rstar = R * Rsol
    
    cs = get_cs(T)
    
    b = G * Mstar /(Rstar * cs**2)
    
    return b

def get_mu(X = 0.7381, Y = 0.2485, Z = 0.0134):
    mu = (2*X + 3/4*Y + 1/2*Z)**-1
    return mu

def get_cs(T):
    '''
    Gets the ideal isothermal sound speed in cgs units asssuming fully ionized with solar metalicity
    
    '''
    
    mu = get_mu()
    
    T = 1e4
    kb = 1.38e-16
    mh = 1.67e-24

    cs = np.sqrt(kb * T/(mu * mh))
    
    return cs

def get_rad(Ncells, rdisk, index):
    '''
    Gets the ra and rb grid, which matches the system from the simulation exactly.
    
    '''
    ra = np.zeros(Ncells+5) # %plus 5 for ghost zones
    rmin = 1
    temp = np.linspace(rmin**(1/index), rdisk**(1/index), (Ncells+3)*2)
    ra[2:Ncells+5]= np.linspace(temp[0], temp[len(temp)-2], Ncells+3)
    
    drindex= ra[3] - ra[2]
    ra[1]=ra[2]-drindex
    ra[0]=ra[1]-drindex
    ra[Ncells+3]=ra[Ncells+2]+drindex
    ra[Ncells+4]=ra[Ncells+3]+drindex
    
    rb=ra + drindex/2
    
    #THESE EXACTLY MATCH THE RA/RB VALUES FROM THE SIMULATIONS
    ra=ra**index
    rb=rb**index
    
    return ra, rb

#def get_bern(u, b, ra, tha, rdisk, gamma, rho, pressure):
#    
#    bern = u.^2/2  - b./ra - b.*ra.^2.*sin(tha).^2./(rdisk.^3.*2) + 1/(gamma-1) .* rho(length(rho)-1)./e(length(e)-1).*e./rho;
