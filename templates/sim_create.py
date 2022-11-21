import numpy as np
import itertools
from astropy.io import ascii
import pdb

'''
sim_create.py

PURPOSE:
    Package that creates matlab jobfiles that for 1D magnetospheric accretion models 
    This script was created to allow the code to be run in parallel on the SCC.

HOW TO USE THIS CODE:
    Change the parameters in the script and run it in python.


NOTES:
    Need to make sure that any variables that are added to the code are also added here.
    Also need to make sure that the variable in the template follow this syntax:
    
    variable = [10];
    
    because this code uses the []'s and the semicolon to grab onto and rewrite variables.

AUTHOR:
    Connor Robinson, connorr@bu.edu
'''

def script():
    ## Set up paths ect.
    path = '/Users/Connor/Desktop/Research/fluid/simcreatetest/'
    tagbase = 'testsimcreate'
    saver = 1
    jobnumstart = 1
    
    ## Switch for running on the cluster
    cluster    = [0] #Turns off progress bar, switches code for different version of matlab
    ## General simulation parameters
    rdisk      = [5, 10] # Outer grid radius
    GAMMA      = [0, 10] # Boct/Bdip
    n          = [2.5, 3.5, 4.5] # Polytropic index
    ## Set up how the boundary condition will vary if variation is required
    timescale  = [1,2,4,6,8] # THIS CODE IS SET UP TO BE IN PHYSICAL UNITS (PERIOD IS IN DAYS)
    amplitude  = [1,3,5]
    delay      = [15]
    ## Stellar parameters
    Mstar      = [0.5 * 2e30] # Mass of the star in kg
    R_sol      = [1.5] # Radius of the star in solar radii
    Tdisk      = [10000]# K # Temperature at the disk due to UV heating
    ## Parameters used for settin up the grid
    Ncells     = [1024] # Number of grid cells
    Nrun       = [1e6] # Number of time-steps
    Ndump      = [250] # Store data every Ndump time-steps
    index      = [8] # grid spacing setup
    ## Parameters that are only important in specific cases (e.g. odd BCs)
    rho_star   = [1e2] # Surface density of the star
    mdot       = [0.1] # Accretion rate
    isothermal = [0] # If this is set to 1, then model is isothermal.
    u_i        = [-1e-5] # Initial velocity for entire simulation (Probably -1 through 0)
    ## Parameters that almost never change
    cs         = [1] # Isothermal sound speed
    rho_disk   = [1] # Density of the disk
    Z          = [0.05]# Height of the wall in stellar radii
    
    ## Boundary conditions
    in_type    = [1]
    in_type    = [1]
    disk_variability = [1]
    
    ## Description of all the boundary conditions:
    ## Inner boundary condition
    # 1 = simple outflow
    # 2 = hard wall hydrostatic
    # 3 = hydrostatic with decay
    
    ## Outer boundary condition
    # 0 = doesn't constrict outflow, simple inflow
    # 1 = simple inflow, constant entropy (adiabatic variability)
    # 2 = set accretion via mdot
    # 3 = hydrostatic (doesn't work well)
    # 4 = simple inflow, constant pressure (doesn't work well)
    
    ## Setting up disk variability
    # 0 = No variability
    # 1 = Sinusoidal with: period = timescale, amplitude
    # 2 = Step function: step occurs at delay, step height = rho0 * amplitude
    # 3 = Top hat function: first step occurs at delay, width = timescale, height = amplitude
    
    #Open up a file and print the parameter names
    f = open(path+tagbase+'-job_params.txt', 'w') 
    f.writelines('jobnum, cluster, rdisk, GAMMA, n, timescale, amplitude, delay, Mstar, R_sol, Tdisk, Ncells, Nrun, Ndump, index, rho_star, mdot, isothermal, u_i, cs, rho_disk, Z, in_type, in_type, disk_variability \n') 
    
    #Write each iteration as a row in the table
    for ind, values in enumerate(itertools.product(cluster, rdisk, GAMMA, n, timescale, amplitude, delay, Mstar, R_sol, Tdisk, Ncells, Nrun, Ndump, index, rho_star, mdot, isothermal, u_i, cs, rho_disk, Z, in_type, in_type, disk_variability)):
        f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
    f.close()
    
    #Open up the table
    table = ascii.read(path+tagbase+'-job_params.txt') 
    
    create(path, tagbase, table[1], table[0])
    
    #def create(cluster, rdisk, GAMMA, n, Mstar, Ncells, Nrun, Ndump, index,rho_star,\
    #    mdot, isothermal, u_i, cs, rho_disk, Z, in_type, out_type, disk_variability,\
    #    timescale, amplitude, delay, R, Tdisk,path, tagbase, saver):
    
def create(path,tagbase,table,names,saver=1, samplepath = ''):
    '''
    Creates the job file + the batch file to run it with
    
    INPUTS:
        path: location of the job parameter list
        tagbase: the associated tag for naming
        table: A row from the table containing all of the parameters
        names: The first row in the table containing all of the parameters names.
    
    OPTIONAL INPUTS:
        saver: Determines if the runs will be saved to .mat files. Probably always on, but can be turned off for testing.
        samplepath: Path to the location of the sample. Default is in this directory.
    
    OUTPUTS:
        Matlab scripts that can be used to run models
    
    '''
    ## Write the matlab file
    #-------------------------------
    #Read in template matlab script
    paramfile = open(samplepath+'run_template.m', 'r')
    fulltext = paramfile.readlines()     # All text in a list of strings
    paramfile.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Replace the dummy parameter in brackets with the parameter from the table
    for i, param in enumerate(names):    
        start = text.find(param + ' = [')+len(param+' = [')
        end = start + len(text[start:].split(']')[0])
        text = text[:start] + str(table[i]) + text[end:]
    
    #Replace the path in the template file
    pathstart = text.find("path = ['") +len("path = ['")
    pathend   = pathstart +len(text[pathstart:].split("']")[0])
    text = text[:pathstart] + path + text[pathend:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(path+'job'+table[0].zfill(4)+'.m', 'w')
    newjob.writelines(outtext)
    newjob.close()
    
    ## Write the batch file
    #-------------------------------
    
    #Read in template batch file
    batchfile = open(samplepath+'batch_template', 'r')
    batchtext = batchfile.readlines()
    batchfile.close()
    
    #Transform text into one long string
    batch = ''.join(batchtext)
    
    #Replace the job number with the one from the table
    bstart = batch.find('job')+len('job')
    bend   = batch.find(';')
    
    batch = batch[:bstart] + table[0].zfill(4) + batch[bend:]
    
    #Turn the text back into something that can be written out
    outbatch = [s + '\n' for s in batch.split('\n')]
    
    #Write out the batch file
    newbatch = open(path+'job'+table[0].zfill(4), 'w')
    newbatch.writelines(outbatch)
    newbatch.close()
    
    
    
