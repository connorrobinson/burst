import numpy as np
import scipy.io

#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
from astropy.table import Table
import os
from astropy.io import fits
from astropy.io import ascii
from sympy.solvers import solve
import sympy as sp
from multiprocessing import Pool
from multiprocessing import cpu_count
from astropy.stats import LombScargle
import itertools
from supersmoother import SuperSmoother
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

from scipy import optimize
from scipy import interpolate
from scipy import signal

from tqdm import tqdm
import math


#from sklearn import gaussian_process as gp
#from sklearn.gaussian_process.kernels import RBF

'''
connector.py

PURPOSE:
    Takes input from the Robinson et al. (2017) accretion simulations and produces
    accretion shock job files for the Calvet et al. (1998) accretion shock RT 
    models (with the Robinson et al. (2018) updates).

NOTES:
    Goal here is to produce an accretion shock model for each timestep
    Only changing input parameters are density and velocity. Also "template" spectra (not that it matters much)
    Probably want it to produce a folder.
    
AUTHOR:
    Connor Robinson, October 5th, 2018
'''

def connector(outname, simulation, Tdisk, DISTANCE, TSTAR, RADIUS, n0, tstart, tend, cadence, \
              runmodel = True,\
              composition = 'solar',\
              outpath = None,\
              templatespectra = '/Users/connor/Dropbox/Research/burst/code/templates/template_spectra.dat',\
              shocktemplate = '/Users/Connor/Dropbox/Research/burst/code/templates/burst_template',\
              runalltemplate = '/Users/connor/Dropbox/Research/burst/code/templates/runall_template',\
              nzeros = 4,\
              DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS',\
              DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES',\
              BASEPATH ='/project/bu-disks/shared/SHOCK/PREPOST/models/',\
              CTFILE ='/project/bu-disks/shared/SHOCK/PREPOST/coolinggrid.txt',\
              COOLTAG ='cooling',\
              CLOUDY ='/projectnb/bu-disks/connorr/cloudy/c17.00/source/cloudy.exe',\
              OPCFILE ='/project/bu-disks/shared/SHOCK/PREPOST/opacitygrid.txt',\
              LOCALPYTHONPATH = None,\
              nthreads = None):
    
    '''
    connector()
    
    PURPOSE:
        Wrapper for creating models from accretion burst simulations
              
    INPUTS:
        outname:[str] Name for the job parameter file 
        simulation:[string] File name for the simulation 
        obj:[str] CTTS to be used as a template for the wavelength grid
        DISTANCE:[float] Distance (in parsecs)
        TSTAR:[float] Temperature of the star (in K)
        n0:[float] Conversion factor between code units and physical units (Typically 3e11)
        tstart:[float] Time at which to start creating simulation files
        tend:[float] Time at which to finish simulation files
        cadence:[float] Cadence between timesteps
    
    OPTIONAL INPUTS:
        runmodel:[bool] True if you want to immediately run the models.
        Composition:[string] Sets the mean molecular weight. Options are 'solar' or 'hydrogen'. Assumes that all of the gas is ionized. NEED TO THINK ABOUT THIS
        outpath:[string] Output location for the code
        shocktemplate:[string] Template file for the burst accretion shock models
        runalltemplate:[str] Template file for the runall. Not necessary to use if running locally. 
        nzeros:[int] Zero padding
        DIRPROG:[str] Path to the program directory for the shock models
        DIRDAT:[str] Path to the data directory for the shock models
        BASEPATH:[str] Path to the base level directory for the pre-post solutions
        CTFILE:[str] Location of the cooling file
        COOLTAG:[str] Name of the cooling file. Generally do not need to change this.
        CLOUDY:[str] Cloudy executable
        OPCFILE:[str] Location of the opacity file for the shock models
        LOCALPYTHONPATH:[str] Python path if running shock models locally. 
        
    AUTHOR:
        Connor Robinson, October 12th, 2018
    '''
    
    #Define constants (in cgs)
    Rsun = 6.96e10
    Msun = 2e33
    G = 6.67e-8
    mh = 1.607e-24
    kb = 1.38066e-16
    
    #Load in the simulation
    sim = scipy.io.loadmat(simulation)
    
    #Set the metallicity
    if composition == 'solar':
        X = 0.7381
        Y = 0.2485
        Z = 0.0134
        mu = (2*X + 3/4*Y + 1/2*Z)**-1
    
    elif compoition == 'hydrogen':
        mu = 0.5
    
    else:
        print('Composition type not found. Current options are: "solar" and "hydrogen". Returning...')
        return np.nan
    
    #Get information from the matlab file
    rho_code = sim['rho_dump'][:,:-1]
    u_code = sim['u_dump'][:,:-1]
    time_code = sim['t_dump'][:-1].flatten()
    
    #Do necessary conversinos from code units to physical units
    rho0 = mu * mh * n0
    cs = np.sqrt(kb * Tdisk/(mu * mh))
    sct = (RADIUS * Rsun/cs) / (24 * 60 * 60) #Sound crossing time in days
    b = sim['M'][0][0]
    rho = rho_code[1,:]*rho0
    u = u_code[1,:]*cs
    time = time_code * sct
    
    #Select timesteps
    tsteps = np.unique([np.argmin(np.abs(t - time)) for t in np.arange(tstart, tend, cadence)])
    
    #Calculate inputs for shock models
    VELOCITY = np.abs(u[tsteps])
    RHO = rho[tsteps]
    BIGF = 0.5 * RHO * VELOCITY**3
    
    MASS = (RADIUS * Rsun) * cs**2 * b/G/Msun
    FILLING = 0.01 #Dummy variable
    SFACTOR = 1.0 #Dummy variable
    
    #Create the models + parameter file
    #Open up a file and print the parameter names
    nstep = len(tsteps)
    jobnum = [str(x).zfill(nzeros) for x in np.arange(nstep)+1]
    names = ['jobnum', 'time', 'DISTANCE', 'MASS', 'RADIO', 'BIGF', 'RHO', 'VELOCITY', 'TSTAR', 'FILLING', 'SFACTOR']
    
    table = Table(np.vstack([jobnum, time[tsteps], np.array([DISTANCE]*nstep).flatten(), np.array([MASS]*nstep).flatten(), \
                             np.array([RADIUS]*nstep).flatten(), BIGF, RHO, VELOCITY, np.array([TSTAR]*nstep).flatten(), np.array([FILLING]*nstep).flatten(), np.array([SFACTOR]*nstep).flatten()]).T,\
                  names = names)
    
    #Create a directory to store models in
    if outpath == None:
        outpath = os.getcwd()+'/'+outname
    
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    if outpath[-1] != '/':
        outpath = outpath + '/'
    
    table.write(outpath+outname+'_shock_params.dat', format = 'ascii', overwrite = True)
    
    for row in table:
        create(row, names, outname, templatespectra, \
        samplefile = shocktemplate,\
        DIRPROG = DIRPROG,\
        DIRDAT = DIRDAT,\
        outpath = outpath,\
        BASEPATH = BASEPATH,\
        CTFILE =CTFILE,\
        COOLTAG = COOLTAG,\
        CLOUDY = CLOUDY,\
        OPCFILE =OPCFILE,\
        LOCALPYTHONPATH = LOCALPYTHONPATH)
    
    #Create the run all script
    create_runall(jobnum[0], jobnum[-1], outpath, outpath = outpath, samplefile = runalltemplate, nzeros = nzeros)
    
    plt.plot(time, np.abs(0.5 * rho * u**3))
    plt.plot(time[tsteps], BIGF, color = 'r', marker = 'o', ls = '')
    
    plt.xlabel(r'$t \, [days]$')
    plt.ylabel(r'$F$')#r'$\dot{M} \, [arb.]$')
    plt.title('Start: '+str(tstart) + 'd, End: '+str(tend)+'d, Cadence: '+str(cadence)+r'd, $N_{steps}$: '+str(len(tsteps)))
    plt.savefig(outpath+outname+'+_timesteps.pdf')
    plt.close()
    
    #Save the model input parameters into something more manageable
    
    sim_data = np.array([sim['M'][0][0],\
            sim['GAMMA'][0][0],\
            sim['rdisk'][0][0],\
            sim['amplitude'][0][0],\
            sim['n'][0][0],\
            sim['delay'][0][0],\
            MASS,\
            RADIUS,\
            cs,\
            sct,\
            mu,\
            Tdisk])
            
    sim_names = np.array([\
                'b',\
                'Gamma',\
                'Rin',\
                'amplitude',\
                'polytropic',\
                'delay',\
                'mass',\
                'radius',\
                'cs',\
                'sct',\
                'mu',\
                'tdisk'])
    
    #Save a table with the simulation parameters
    simtable = Table(sim_data, names = sim_names)
    ascii.write(simtable, outpath+outname+'_sim_params.dat', overwrite = True)
    
    #Run the models if requested
    if runmodel:
        #Move to the code location
        #os.system('cd '+outpath)
        os.chdir(outpath)
        #Construct the name of the job files
        jobnames = [outpath + 'job' + j for j in jobnum]
        
        #If nthreads is not specified, use number of cores - 1
        if nthreads == None:
            nthreads = cpu_count() - 1
        
        #Run the models
        pool = Pool(nthreads)
        
        pool.map(run_model, jobnames)
        
    
def run_model(job):
    '''
    
    connector.run_model()
    
    PURPOSE:
        Ancillary function to run jobs via csh with pool
    
    AUTHOR:
        Connor Robinson, January 29th, 2019
    
    '''
    print('Running job'+job)
    os.system('csh '+job)

def phase1(tagbase, clusterpath, savepath, \
            GAMMA, RDISK, AMPLITUDE, MSTAR, RSTAR, TSTAR, DISTANCE, \
            n0, tstart, tend, cadence, Nrun, timescale,\
            out_type, disk_variability, in_type, \
            wttsname, jobid, p4path, clusterp4path,\
            u_i = -1e-5, Ncells = 1024, delay = 15, Ndump = 125, index = 8, cluster = 1, Tdisk = 10000, n = 10000, isothermal = 1,\
            cs = 1, rho_disk = 1, Z = 0.05, jobnumstart =1, rho_star = 1e2, mdot = 0.1,\
            phi0 = 0, alpha = None, ftrunc = None, width = None, incs = [0,10,20,30,40,50,60,70,80,90], \
            pt = 0.2, ell_q = 0.3, ell_m = 4/24, nzeros = 4, boundary_file = None, \
            Fmax = 0, Fmin = 0, \
            localtemplatepath = '/Users/Connor/Dropbox/Research/burst/code/templates/',\
            clustertemplatepath = '/projectnb/bu-disks/connorr/burst/code/templates/',\
            localpathfile = '/Users/Connor/Dropbox/Research/burst/code/paths/local_paths.dat',\
            clusterpathfile = '/projectnb/bu-disks/connorr/burst/code/paths/scc_paths.dat',\
            ):
    '''
    
    PURPOSE:
        Contains all the code to run the simulations and the framework to hold results and plots.
    
    INPUTS:
        tagbase:[str] Name given to model
        clusterpath:[str] Location on cluster for running models
        savepath:[str] Location to save files
        GAMMA:[float] Boct/Bdip
        RDISK:[float] Inner disk radius (in stellar radii)
        AMPLITUDE:[float] Size of changes for variabile BCs. For Kolmogorav, sets Mach number RMS
        MSTAR:[float] Mass of the star in solar masses
        RSTAR:[float] Radius of the star in solar radii
        TSTAR:[float] Tempature of the star
        DISTANCE:[float] Distance to star in pc
        n0:[float] Density of column at disk in cm^-3
        tstart:[float] Time in days to start generating shock models
        tend:[float] Time in days to stop generating shock models
        cadence:[float] Cadence between shock models
        Nrun:[int] Number of time steps in simulation
        timescale:[float] Timescale in days for driving functions. Not used for turbulence flow. 
            
        out_type:[choices]
            0 = doesn't constrict outflow, simple inflow
            1 = simple inflow, constant entropy (adiabatic variability)
            2 = set accretion via mdot
            3 = hydrostatic (doesn't work well)
            4 = simple inflow, constant pressure (doesn't work well)
        
        disk_variability:[choices] 
            0 = No variability
            1 = Sinusoidal with: period = timescale, amplitude
            2 = Step function: step occurs at delay, step height = rho0 * amplitude
            3 = Top hat function: first step occurs at delay, width = timescale, height = amplitude
            5 = Kolmogorav turbulence spectrum, set either by parameters above or by a fixed upper/lower frequency bound
        
        in_type:[choices] 
            1 = simple outflow
            2 = hard wall hydrostatic
            3 = hydrostatic with decay
        
        wttsname:
        
    OPTIONAL INPUTS:
        u_i:[float] Initialization velocity of the simulation. 
        Ncells:[int] Number of grid cells
        delay[float] Delay to start driving forces. Not used for turbulent flow
        Ndump:[int] Store data every Ndump time-steps
        index:[float]  grid spacing setup
        cluster:[bool] Turns off progress bar, switches code for different version of matlab
        Tdisk:[float] Temperature at the disk due to UV heating. Typical is 10000K
        n:[float] Polytropic index. Ignored if isothermal is True
        isothermal:[bool] Turn on isothermal version of code
        cs:[float] Code velocity units. Never a reason to change this from 1 as far as I can see.
        rho_disk:[float] Code density units. Never a reason to change this from 1. Changing it can cause numerical issues.
        Z:[float] Height above midplane to start simulation. 0.05 is typical.
        jobnumstart:[float] Job number to start at. 
        rho_star:[float] Density of star. Used in some inner bc's, but is rare. 
        mdot:[float] Sets mass accretion rate for some BCs, but is rarely used.
        phi0:[int] Initial rotational phase. Default is 0. 
        alpha:[int] Size of the spot in degrees. Default is 5
        incs:[float arr] Inclinations to run rode at. Default is [0,10,20,30,40,50,60,70,80,90]
        pt:[float] Threshold at which to select peak from. 
        ell_q:[float] Correlation length scale for GP when determining Q, Default is 0.3 (in phase space)
        ell_m:[float]  Correlation length scale for GP when determining M, Default is 4 hours.
        nzeros:[float] Zero padding. Default is 4. 
        boundary_file:[str] Name of a file containing the boundary condition information. Used if using the same turbulence law for multiple models
        
        localtemplatepath:[str] Location of local template files
        clustertemplatepath:[str] Location of template files on the cluster
        localpathfile:[str] Location of local paths file
        clusterpathfile:[str] Location of cluster paths file
        
    AUTHOR:
        Connor Robinson, June 7th, 2019
    
    '''
    
    #Set some of the parameters as lists for creation of matlabfile
    GAMMA      = [GAMMA] # Boct/Bdip
    rdisk      = [RDISK] # Outer grid radius (in stellar radii)
    amplitude  = [AMPLITUDE] # Size of changes. For Kolmogorav, sets Mach number RMS
    Mstar      = [MSTAR * 2e30] # Mass of the star in kg
    R_sol      = [RSTAR] # Radius of the star in solar radii
    model      = tagbase
    Nrun       = [Nrun] # Number of time-steps
    numin      = [Fmin]
    numax      = [Fmax]
    out_type   = [out_type]
    disk_variability = [disk_variability]
    delay      = [delay] # days
    timescale  = [timescale] # days
    in_type    = [in_type] #Inflow boundary condition
    rho_star   = [rho_star] # Surface density of the star
    mdot       = [mdot] # Accretion rate
    u_i        = [u_i] # Initial velocity for entire simulation (Probably -1 through 0)
    Ncells     = [Ncells] # Number of grid cells
    Ndump      = [Ndump] # Store data every Ndump time-steps
    index      = [index] # grid spacing setup
    cluster    = [cluster] #Turns off progress bar, switches code for different version of matlab
    Tdisk      = [Tdisk]# K, Temperature at the disk due to UV heating
    n          = [n] # Polytropic index
    isothermal = [isothermal] # If this is set to 1, then model is isothermal. For now, models should be isothermal
    cs         = [cs] # Isothermal sound speed
    rho_disk   = [rho_disk] # Density of the disk
    Z          = [Z] # Height of the wall in stellar radii
    
    # #############################################
    # Should not need to change anything below here
    # #############################################
    
    #PHASE 1 -- Initialization 
    ##############################################
    #Build directories for storing simulations
    simpath     = savepath + 'simulation/'
    modpath     = savepath + 'models/'
    codepath    = savepath + 'code/'
    supportpath = savepath + 'support/'
    datapath    = savepath + 'data/'
    resultspath = savepath + 'results/'
    figpath     = savepath + 'figs/'
    p3path      = modpath + model + '/'
    
    os.system('mkdir ' + savepath)       #Creates the top level file for everything. If already created, does not do anything bad.
    os.system('mkdir ' + simpath)        # Stores simulation + run files for cluster
    os.system('mkdir ' + modpath)        # Stores shock models
    os.system('mkdir ' + codepath)       # Stores phase 2 code + any additional analysis codes
    os.system('mkdir ' + supportpath)    # Stores parameter tables, ect.
    os.system('mkdir ' + datapath)       # Empty for now, but can store things like lightcurves
    os.system('mkdir ' + resultspath)    # Empty for now, but will fill with science :)
    os.system('mkdir ' + figpath)        # Empty for now, but will fill with plots
    os.system('mkdir ' + p3path)         # Directory for shock models
    
    # PHASE 2 PREP -- Running the fluid simulation
    ##############################################
    
    ## Create 1D velocities based on the Kolmogorav turbulence spectrum (if appropriate for outer BC)
    if out_type == [5] and boundary_file == None and Fmin == 0 and Fmax ==  0:
        boundary_file = create_kolmogorav(simpath, clusterpath, tagbase, R_sol[0], rdisk[0], Tdisk[0], amplitude[0], rho_disk[0], tend, \
                                          isothermal = True, Fmax = 2.0/(60*60), n = None)
    
    if out_type == [5] and boundary_file == None and Fmin != 0 and Fmax != 0:
        boundary_file = create_kolmogorav_fixed(Fmax, Fmin, simpath, clusterpath, tagbase, R_sol[0], Tdisk[0], amplitude[0], rho_disk[0], tend, isothermal = True, n = None)
    
    ## Create the phase 1 table to be used in the simulation
    f = open(supportpath+tagbase+'_sim_params.txt', 'w') 
    f.writelines('jobnum, cluster, rdisk, GAMMA, n, timescale, amplitude, delay, Mstar, R_sol, Tdisk, Fmin, Fmax, Ncells, Nrun, Ndump, index, rho_star, mdot, isothermal, u_i, cs, rho_disk, Z, in_type, out_type, disk_variability \n') 
    
    #Write each iteration as a row in the table
    for ind, values in enumerate(itertools.product(cluster, rdisk, GAMMA, n, timescale, amplitude, delay, Mstar, R_sol, Tdisk, numin, numax, Ncells, Nrun, Ndump, index, rho_star, mdot, isothermal, u_i, cs, rho_disk, Z, in_type, out_type, disk_variability)):
        f.writelines(str(ind+jobnumstart)+', '+ str(values)[1:-1]+ '\n')
    f.close()
    
    #Open up the table and create the job files
    table = ascii.read(supportpath+tagbase+'_sim_params.txt')
    
    for line in table:
        create_sim(clusterpath, simpath, tagbase, line, samplepath = localtemplatepath, boundary_file = boundary_file)
    
    # PHASE 3 PREP -- Running the shock model
    ##############################################
    
    #Define p4path
    
    #Create a larger table using the phase 2. Create a copy in simulation files for easy transfer
    create_p3table(supportpath+tagbase+'_sim_params.txt', supportpath, model, DISTANCE, TSTAR, n0, tstart, tend, cadence)
    create_p3table(supportpath+tagbase+'_sim_params.txt', simpath, model, DISTANCE, TSTAR, n0, tstart, tend, cadence)
    
    #Define location of the local table and the remote copy
    localp3table = supportpath + model + '_params.dat'
    clusterp3table = clusterpath + model + '_params.dat'
    
    #Define the simulation name ASSUMES 2 THINGS: ISOTHERMAL SOLUTION AND ONLY 1 SIMULATION
    simulation = simpath + tagbase+str(jobnumstart).zfill(3)+'--iso.mat'
    clustersimulation = clusterpath + tagbase+str(jobnumstart).zfill(3)+'--iso.mat'
    
    #Create two versions of the phase 3 code, one for local running and one for the cluster (placed with the simulation file)
    create_p3code(model, localp3table, codepath, localtemplatepath, simulation, p3path, localpathfile, jobid, p4path, remote = False)
    create_p3code(model, clusterp3table, simpath, localtemplatepath, clustersimulation, clusterpath+model+'/', clusterpathfile, jobid, clusterp4path, remote = True)
    
    # PHASE 4 PREP -- Creating the object and analyzing the results
    ##############################################
    
    #Construct paths
    localmodelpath = modpath+model+'/'
    clustermodelpath = clusterpath+model+'/'
    
    localparamfile = supportpath+tagbase+'_sim_params.txt'
    clusterparamfile = clusterpath+model+'/'+model+'_sim_params.dat'
    
    localshockparamfile = modpath+model+'/'+model+'_shock_params.dat'
    clustershockparamfile = clusterpath+model+'/'+model+'_shock_params.dat'
    
    #Add wttsparam file to paths
    band = 'V'
    
    if (alpha != None) and (ftrunc == None) and (width == None):
        create_p4code(model, localp3table, localshockparamfile,  wttsname, localmodelpath, localmodelpath, localtemplatepath, figpath, localpathfile, band,\
                          phi0 = 0, alpha = alpha, incs = [0,10,20,30,40,50,60,70,80,90], \
                          pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = False)
        
        create_p4code(model, clusterp3table, clustershockparamfile, wttsname, clustermodelpath, p4path, localtemplatepath, clustermodelpath, clusterpathfile, band,\
                          phi0 = 0, alpha = alpha, incs = [0,10,20,30,40,50,60,70,80,90], \
                          pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = clusterp4path)
    
    elif (alpha == None) and (ftrunc != None) and (width != None):
        create_p4strip_code(model, localp3table, localshockparamfile,  wttsname, localmodelpath, localmodelpath, localtemplatepath, figpath, localpathfile, band,\
                          phi0 = 0, width = width, ftrunc = ftrunc, incs = [0,10,20,30,40,50,60,70,80,90], \
                          pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = False)
        
        create_p4strip_code(model, clusterp3table, clustershockparamfile, wttsname, clustermodelpath, p4path, localtemplatepath, clustermodelpath, clusterpathfile, band,\
                          phi0 = 0, width = width, ftrunc = ftrunc, incs = [0,10,20,30,40,50,60,70,80,90], \
                          pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = clusterp4path)
    
    else:
        raise ValueError('Conflicting hot spot choices. \n Set either alpha for spherical cap model or width & ftrunc for strips.')
    
def create_p3code(model, p3table, codepath, templatepath, simulation, outpath, pathfile, \
                  jobid, p4path, remote = False):
    '''
    
    connector.create_p3code()
    
    PURPOSE:
        Generates the code for the initial parameters for running shock models
    
    INPUTS:
        model:[str] Name of the model
        p3table:[str] Location of the table file containing all the parameters
        codepath:[str] Location of where to save the code after running
        templatepath:[str] Location of the template files
        simulation:[str] Name of the simulation file
        outpath:[string] Output location for the code (used by connector.connector)
        pathfile:[str] Location file containing relevant paths
        jobid:[str] Name for phase 4
        p4path:[str] Location to place the p4 job file.
    
    AUTHOR:
        Connor Robinson, January 15th, 2019
    '''
    
    #Copy the template
    codefile = codepath+model+'_phase3.py'
    
    if remote:
        os.system('cp ' + templatepath + 'runshock_template_remote.py ' + codefile)
    else:
        os.system('cp ' + templatepath + 'runshock_template.py ' + codefile)
    
    #Open the template and replace the necessary parts of the file
    p3file = open(codefile,  'r')
    fulltext = p3file.readlines()     # All text in a list of strings
    p3file.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Set up parts to replace
    replace = {'model':model, 'simulation':simulation, 'tablefile':p3table, 'outpath':outpath, 'pathfile':pathfile, 'jobid':jobid, 'p4path':p4path}
    
    #Replace each part
    for rep in replace.keys():
        start = text.find(rep + " = '")+len(rep+" = '")
        if start - len(rep+" = '") == -1:
            continue
        else:
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + replace[rep] + text[end:]
            
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(codefile, 'w')
    newjob.writelines(outtext)
    newjob.close()
    

def create_p4code(model, tablefile, shockparamfile, wttsname, modelpath, p4path, templatepath, figpath, pathfile, band, \
                  phi0 = 0, alpha = 5, incs = [0,10,20,30,40,50,60,70,80,90], \
                  pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = None):
    '''
    connector.create_p4code
    
    PURPOSE:
        Creates the code + jobfile for combining the models into an object and calculating Q and M for each inclination. 
    
    INPUTS:
        model:[str] Name of the model
        tablefile:[str] Name of the tablefile assocaited with the shock models
    
    AUTHOR:
        Connor Robinson, June 19th, 2019
    
    '''
    #Copy the template
    
    
    if clusterp4path:
        codefile = p4path+model+'_phase4.py'
        os.system('cp ' + templatepath + 'create_obj_template_remote.py ' + codefile)
        p4pathsave = clusterp4path
    else:
        codefile = p4path+model+'_phase4.py'
        os.system('cp ' + templatepath + 'create_obj_template.py ' + codefile)
        p4pathsave = p4path
    
    #Open the template and replace the necessary parts of the file
    p4file = open(codefile,  'r')
    fulltext = p4file.readlines()     # All text in a list of strings
    p4file.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Set up parts to replace
    replace_str = {'model':model, 'wttsname':wttsname, 'tablefile':tablefile, 'shockparamfile':shockparamfile,\
                   'modelpath':modelpath, 'pathfile':pathfile, 'figpath':figpath, 'p4path':p4pathsave, 'band':band}
    
    replace_flt = {'phi0':phi0, 'alpha':alpha, 'incs':incs, 'pt':pt, 'ell_q':ell_q, 'ell_m':ell_m, 'fracerr':fracerr, 'nzeros':nzeros}
    
    #Replace each part
    for rep in replace_str.keys():
        start = text.find(rep + " = '")+len(rep+" = '")
        end = start + len(text[start:].split("'")[0])
        text = text[:start] + replace_str[rep] + text[end:]
        
    for rep in replace_flt.keys():
        start = text.find(rep + " = ")+len(rep+" = ")
        end = start + len(text[start:].split("\n")[0])
        text = text[:start] + str(replace_flt[rep]) + text[end:]
        
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(codefile, 'w')
    newjob.writelines(outtext)
    newjob.close()

def create_p4strip_code(model, tablefile, shockparamfile, wttsname, modelpath, p4path, templatepath, figpath, pathfile, band, \
                  phi0 = 0, ftrunc = 0.5, width = 30, incs = [0,10,20,30,40,50,60,70,80,90], \
                  pt = 0.2, ell_q = 0.3, ell_m = 4/24, fracerr = 0.005, nzeros = 4, clusterp4path = None):
    '''
    connector.create_p4code
    
    PURPOSE:
        Creates the code + jobfile for combining the models into an object and calculating Q and M for each inclination. 
    
    INPUTS:
        model:[str] Name of the model
        tablefile:[str] Name of the tablefile assocaited with the shock models
    
    AUTHOR:
        Connor Robinson, June 19th, 2019
    
    '''
    #Copy the template
    if clusterp4path:
        codefile = p4path+model+'_phase4.py'
        os.system('cp ' + templatepath + 'create_obj_strip_template_remote.py ' + codefile)
        p4pathsave = clusterp4path
    else:
        codefile = p4path+model+'_phase4.py'
        os.system('cp ' + templatepath + 'create_obj_strip_template.py ' + codefile)
        p4pathsave = p4path
    
    #Open the template and replace the necessary parts of the file
    p4file = open(codefile,  'r')
    fulltext = p4file.readlines()     # All text in a list of strings
    p4file.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Set up parts to replace
    replace_str = {'model':model, 'wttsname':wttsname, 'tablefile':tablefile, 'shockparamfile':shockparamfile,\
                   'modelpath':modelpath, 'pathfile':pathfile, 'figpath':figpath, 'p4path':p4pathsave, 'band':band}
    
    replace_flt = {'phi0':phi0, 'ftrunc':ftrunc, 'width':width, 'incs':incs, 'pt':pt, 'ell_q':ell_q, 'ell_m':ell_m, 'fracerr':fracerr, 'nzeros':nzeros}
    
    #Replace each part
    for rep in replace_str.keys():
        start = text.find(rep + " = '")+len(rep+" = '")
        end = start + len(text[start:].split("'")[0])
        text = text[:start] + replace_str[rep] + text[end:]
        
    for rep in replace_flt.keys():
        start = text.find(rep + " = ")+len(rep+" = ")
        end = start + len(text[start:].split("\n")[0])
        text = text[:start] + str(replace_flt[rep]) + text[end:]
        
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(codefile, 'w')
    newjob.writelines(outtext)
    newjob.close()

    


def create_p4job(jobid, p4path, model, template_p4job):
    '''
    
    connector.create_p4job
    
    PURPOSE:
        Creates the p4 job file.
    
    INPUTS:
        p4path:[str] Location to create the job file
        model:[str]
        templatepath:[str] 
    
    AUTHOR:
        Connor Robinson, June 19th, 2019
    
    '''
    
    #Copy the template
    codefile = p4path+'p4_'+ jobid
    os.system('cp ' + template_p4job + ' ' + codefile)
    
    #Open the template and replace the necessary parts of the file
    temp = open(codefile,  'r')
    fulltext = temp.readlines()     # All text in a list of strings
    temp.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Set up parts to replace
    replace = {'p4path':p4path, 'model':model}
    
    #Replace each part
    for rep in replace.keys():
        start = text.find(rep + " = '")+len(rep+" = '")
        end = start + len(text[start:].split("'")[0]) 
        text = text[:start] + replace[rep] + text[end:]
    
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(codefile, 'w')
    newjob.writelines(outtext)
    newjob.close()



def create(row, names, NAME, wttsfile, \
    samplefile = '/Users/connor/Dropbox/Research/burst/code/templates/burst_template',\
    DIRPROG = '/project/bu-disks/shared/shockmodels/PROGRAMS',\
    DIRDAT = '/project/bu-disks/shared/shockmodels/DATAFILES',\
    outpath = '',\
    BASEPATH ='/project/bu-disks/shared/SHOCK/PREPOST/models/',\
    CTFILE ='/project/bu-disks/shared/SHOCK/PREPOST/models/coolinggrid.txt',\
    COOLTAG ='cooling',\
    CLOUDY ='/projectnb/bu-disks/connorr/cloudy/c17.00/source/cloudy.exe',\
    OPCFILE ='/project/bu-disks/shared/SHOCK/PREPOST/models/opacitygrid.txt',\
    LOCALPYTHONPATH = None,\
    nzeros=3):
    
    '''
    shock.create
    
    PURPOSE:
        Creates the shock job file
    
    INPUTS:
        path: location of the job parameter list
        row: A row from the table containing all of the parameters
        names: The first row in the table containing all of the parameters names.
        NAME: name associated with the model
    
    
    OPTIONAL INPUTS:
        samplepath: [String] Path to the location of the sample. Default is in this directory.
        nzeros: [Int] Zero padding in the job number, default is 3
        DIRPORG/DIRDAT: [String] Paths to where the shockmodels themselves live
        outpath: [String] Path to where the files will be written. Default is the current directory.
        BASEPATH: [String] Path to the top level directory containing the cloudy models
        CTFILE: [String] File containing the cooling table
        COOLTAG: [String] Name associated with files in the cooling table
        CLOUDY: [String] Path + name of the cloudy executable file
        OPCFILE:[String] File containing the opacity information
        LOCALPYTHONPATH:[string] 
        
    OUTPUTS:
        Batch file containing the necessary code to the run the model
    
    '''
    ## Write the batch file
    #-------------------------------
    paramfile = open(samplefile,  'r')
    fulltext = paramfile.readlines()     # All text in a list of strings
    paramfile.close()
    
    #Transform the text into one long string
    text = ''.join(fulltext)
    
    #Replace the dummy parameter in brackets with the parameter from the table
    for i, param in enumerate(names):
        
        if param == 'jobnum':
            start = text.find(param + "='")+len(param+"='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + str(row[i]).zfill(nzeros) + text[end:]
            
        elif param == 'time':
            continue
        else:
            start = text.find(param + "='")+len(param+"='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + str(row[i]) + text[end:]
    
    #Replace the WTTS file
    start = text.find("set filewtts=")+len("set filewtts=")
    end = start +len(text[start:].split("\n")[0])
    text = text[:start] + NAME+'_template_' + text[end:]
    
    #Replace the template (This is a work around to avoid changing collate. Gross.)
    start = text.find("set template='")+len("set filewtts='")
    end = start +len(text[start:].split("'")[0])
    text = text[:start] + wttsfile + text[end:]
    
    #Set the name of the file
    start = text.find("NAME='")+len("NAME='")
    end = start + len(text[start:].split("'")[0])
    text = text[:start] + NAME+str(row[0]).zfill(nzeros) + text[end:]
    
    #Set the paths
    start = text.find('DIRPROG=')+len('DIRPROG=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRPROG + text[end:]
    
    start = text.find('DIRDAT=')+len('DIRDAT=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + DIRDAT + text[end:]
    
    #Set up the cloudy stuff
    start = text.find('set BASEPATH=')+len('set BASEPATH=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + BASEPATH + "'"+ text[end:]
    
    start = text.find('set CTFILE=')+len('set CTFILE=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + CTFILE + "'" + text[end:]
    
    start = text.find('set COOLTAG=')+len('set COOLTAG=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start]+ "'" + COOLTAG + "'" + text[end:]
    
    start = text.find('set CLOUDY=')+len('set CLOUDY=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + "'" + CLOUDY + "'" + text[end:]
    
    start = text.find('set OPCFILE=')+len('set OPCFILE=')
    end = start + len(text[start:].split('\n')[0])
    text = text[:start] + "'" + OPCFILE + "'" + text[end:]
    
    #Fix python if local job
    if LOCALPYTHONPATH != None:
        
        start = text.find('python -c "import prepost_burst;')
        end = start + len(text[start:].split(';')[0]) 
        text = text[:start]+ LOCALPYTHONPATH + 'w -c "import prepost_burst'+ text[end:]
        
        start = text.find('python -c "import collate;')
        end = start + len(text[start:].split(';')[0]) 
        text = text[:start]+ LOCALPYTHONPATH + 'w -c "import collate'+ text[end:]
        
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(outpath+'job'+str(row[0]).zfill(nzeros), 'w')
    newjob.writelines(outtext)
    newjob.close()

def create_sim(simpath, savepath, tagbase, line, saver=1, samplepath = '', boundary_file = None):
    '''
    Creates the job file + the batch file to run it with
    
    INPUTS:
        simpath: Location on the cluster where this will be run
        savepath: Location of where the simulations will initially be saved
        tagbase: the associated tag for naming
        line: A row from the table containing all of the parameters
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
    
    names = line.colnames
    
    
    #Replace the dummy parameter in brackets with the parameter from the table
    for i, param in enumerate(names):
        
        #Handle the cases that are not changed in the run file
        if param == 'Fmin' or param == 'Fmax':
            continue
        else:
            start = text.find(param + ' = [')+len(param+' = [')
            end = start + len(text[start:].split(']')[0])
            text = text[:start] + str(line[param]) + text[end:]
    
    #Replace the path in the template file
    pathstart = text.find("path = ['") +len("path = ['")
    pathend   = pathstart +len(text[pathstart:].split("']")[0])
    text = text[:pathstart] + simpath + text[pathend:]
    
    #Replace the tagbase in the template file
    pathstart = text.find("tagbase = ['") +len("tagbase = ['")
    pathend   = pathstart +len(text[pathstart:].split("']")[0])
    text = text[:pathstart] + tagbase + text[pathend:]
    
    #Replace the boundary file if it exists
    if boundary_file != None:
        pathstart = text.find("boundary_file = '") +len("boundary_file = '")
        pathend   = pathstart +len(text[pathstart:].split("'")[0])
        text = text[:pathstart] + boundary_file + text[pathend:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the job file
    newjob = open(savepath+'job'+str(line['jobnum']).zfill(4)+'.m', 'w')
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
    
    batch = batch[:bstart] + str(line['jobnum']).zfill(4) + batch[bend:]
    
    #Replace the tagbase to create the phase 3 file
    tagstart = batch.find("set tagbase = '")+len("set tagbase = '")
    tagend   = tagstart +len(batch[tagstart:].split("'")[0])
    
    batch = batch[:tagstart] + tagbase + batch[tagend:]
    
    #Turn the text back into something that can be written out
    outbatch = [s + '\n' for s in batch.split('\n')]
    
    #Write out the batch file
    newbatch = open(savepath+'job'+str(line['jobnum']).zfill(4), 'w')
    newbatch.writelines(outbatch)
    newbatch.close()

def create_runall(jobstart, jobend, clusterpath, outpath = '', samplefile = 'runall_template', nzeros = 4, p4 = False):
    '''
    shock.create_runall()
    
    INPUTS:
        jobstart: [int] First job file in grid
        jobsend: [int] Last job file in grid
    
    OPTIONAL INPUTS:
        samplepath: Path to where the runall_template file is located. Default is the current directory.
        p4:[bool] If true, replaces the job name with 'p4_' for runnign p4 grids
        
    '''
    #Write the runall script
    runallfile = open(samplefile, 'r')
    fulltext = runallfile.readlines()     # All text in a list of strings
    runallfile.close()
    
    #Turn it into one large string
    text = ''.join(fulltext)
    
    #Replace the path
    start = text.find('cd ')+len('cd ')
    end = start +len(text[start:].split('\n')[0])
    text = text[:start] + clusterpath + text[end:]
    
    #Replace the jobstart
    start = text.find('#qsub -t ')+len('#qsub -t ')
    end = start +len(text[start:].split('-')[0])
    text = text[:start] + str(int(jobstart)) + text[end:]
    
    #Replace the job end
    start = text.find('#qsub -t '+str(int(jobstart))+'-')+len('#qsub -t '+str(int(jobstart))+'-')
    end = start +len(text[start:].split(' runall.csh')[0])
    text = text[:start] + str(int(jobend)) + text[end:]
    
    #Replace nzeros
    start = text.find('job%0')+len('job%0')
    end = start +len(text[start:].split('d" $SGE_TASK_ID')[0])
    text = text[:start] + str(int(nzeros)) + text[end:]
    
    #If p4, replace the job name
    if p4:
        start = text.find('printf "')+len('printf "')
        end = start + len(text[start:].split('%')[0])
        text = text[:start] + 'p4_' + text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the runall file
    newrunall = open(outpath+'runall.csh', 'w')
    newrunall.writelines(outtext)
    newrunall.close()

def create_p3table(p2tablefile, savepath, model, dist, teff, n0, tstart, tend, cadence):
    '''
    
    creat_p3table()
    
    PURPOSE:
        Takes in variables necessary to run shock + table from Phase 2 to create a table with all of the information for Phase 3.
    
    INPUTS:
        
    AUTHOR:
        Connor Robinson
    '''
    
    table = ascii.read(p2tablefile)
    
    table['model'] = model
    table['dist'] = dist
    table['teff'] = teff
    table['n0'] = n0
    table['tstart'] = tstart
    table['tend'] = tend
    table['cadence'] = cadence
    
    ascii.write(table, savepath + model + '_params.dat', overwrite = True)
    
    return table

def kolmogorav(N, Fmin, Fmax):
    '''
    connector.kolmogorav()
    
    PURPOSE:
        Generates 1D velocity information based on the Kolmogorav turbulence spectrum.
        This will be used as a boundary condition for the simulations.
    
    INPUTS:
        N: Number of samples in PSD
        Fmin: Minimum frequency (in Hz)
        Fmax: Maxmimum frequency (in Hz)
    
    OUTPUTS:
        T:[float array] Time in days
        sig:[float array] Real part of the turbulent velocity. 
    
    AUTHOR:
        Connor Robinson, January 29th, 2019
    '''
    #Step 1:  Set PSD
    # Kolmogorav: F(w) \propto w**(-5/3)
    
    #Step 2: Set number of samples in PSD
    M = int(N/2 + 1)
    
    #Step 3.5: Construct PSD
    w = 10**np.linspace(Fmin, Fmax, int(M))
    PSD = w**(-5/3)
    
    #Step 4: Convert PSD to amplitude
    A = np.sqrt(2 * PSD)
    
    #Step 5: Assign random phase
    phi = np.random.uniform(low = 0, high = 2*np.pi, size = int(M))
    
    #Step 6: Convert to frequency domain signal
    Z = A * np.exp(1j*phi)
    
    #Step 7: Apply the FFT
    S = np.fft.ifft(Z, n = N)
    
    sig = S.real
    test = np.abs(np.fft.fft(S))
    
    delT = 1/(2 * 10**Fmax)
    Tmax = (N-1) * delT
    
    day = 24 * 60 * 60
    
    T = np.linspace(0,Tmax,N)
    
    return T/day, sig

def create_kolmogorav_fixed(Fmax, Fmin, simpath, clusterpath, tagbase, Rstar, Tdisk, amplitude, rho_disk, time, isothermal = True, n = None):
    '''
    create_kolmogorav_fixed
    
    PURPOSE:
        Wrapper to run the code that creates Kolmogorav turbulence for the simulations
        In this version the upper and lower frequency bounds are fixed.
        
    INPUTS:
        Fmax:[float] Upper frequency limit.
        Fmin:[float] Lower frequency limit.
        Rstar:[float] Stellar radius in solar radii
        Tdisk:[float] Temperature of the upper atmosphere of disk. Typically 10000K
        amplitude:[float] RMS of the Mach number of the turbulence. Probably best lower than 1 to avoid supersonic turbulence.
        rho_disk[float] Density of the disk in code units. Typically 1. 
        time:[float] Amount of time to create (will create 10x this amount to be safe)
    
    OPTIONAL INPUTS
        isothermal:[bool] Sets the gas behavior. If false, need to set value for adiabatic index n.
        Fmax:[float] Maximum frequency in turbulence spectrum. Default is 2 hours.
        n:[float] Adiabatic index. gamma = (n+1)/n
    
    AUTHOR:
        Connor Robinson, May 17th, 2019
    '''
    
    #Define constants
    kb = 1.38e-23 # J/K
    mp = 1.67e-27 # kg
    
    #Assume solar metalicity
    Xm = 0.7381
    Ym = 0.2485
    Zm = 0.0134
    mu = (2*Xm + 3/4*Ym + 1/2*Zm)**(-1)
    
    #Constants
    G = 6.67e-11 # mks
    Rsun = 6.95508e8 #m
    Msun = 2e30 #kg
    
    #Convert time into code units
    if isothermal:
        gamma = 1
    else:
        gamma = (n+1)/n
    
    #Set stellar parameters
    cs = np.sqrt(gamma * kb * Tdisk / (mu * mp))
    
    #Calculate necessary number of cells. Increase by a factor of 10 to be safe.
    N = 10 * np.ceil(2 * 10**Fmax * (time * 24 * 60 * 60)) + 1 
    
    #Generate the turbulence spectrum
    T, sig = kolmogorav(N, Fmin, Fmax)
    
    #Scale the velocity output by the sound speed and the amplitude of the variations
    if len(np.atleast_1d(amplitude)) > 1:
        
        boundary_file = []
        
        for i, a in enumerate(amplitude):
            sig_cs = sig/cs
            rms = np.std(sig_cs)
            scale_sig = sig_cs * (a/rms)
            
            T_code = convert_time(T, Rstar, Tdisk, gamma, mu = mu)
            
            #Set the density equal to the disk density. 
            rho = np.zeros(len(T)) + rho_disk
            
            #Save it
            np.savetxt(simpath+tagbase+'_kolmogorav_'+str(i+1)+'.dat', np.vstack([T_code, scale_sig, rho]).T)
            boundary_file.append(clusterpath+tagbase+'_kolmogorav_'+str(i+1)+'.dat')
        
        boundary_file = np.array(boundary_file)
        return boundary_file
        
    else:
        sig_cs = sig/cs
        rms = np.std(sig_cs)
        scale_sig = sig_cs * (amplitude/rms)
        
        T_code = convert_time(T, Rstar, Tdisk, gamma, mu = mu)
        
        #Set the density equal to the disk density. 
        rho = np.zeros(len(T)) + rho_disk
        
        #Save it
        np.savetxt(simpath+tagbase+'_kolmogorav.dat', np.vstack([T_code, scale_sig, rho]).T)
        
        boundary_file = clusterpath+tagbase+'_kolmogorav.dat'
        
        return boundary_file
    
    

def create_kolmogorav(simpath, clusterpath, tagbase, Rstar, Rdisk, Tdisk, amplitude, rho_disk, time, isothermal = True, Fmax = 2.0/(60*60), n = None):
    '''
    create_kolmogorav
    
    PURPOSE:
        Wrapper to run the code that creates Kolmogorav turbulence for the simulations
        The minimum frequency is set to be the sound speed over the inner disk circumference
        
    INPUTS:
        Rstar:[float] Stellar radius in solar radii
        Rdisk:[float] Inner disk radius in solar radii
        Tdisk:[float] Temperature of the upper atmosphere of disk. Typically 10000K
        amplitude:[float] RMS of the Mach number of the turbulence. Probably best lower than 1 to avoid supersonic turbulence.
        rho_disk[float] Density of the disk in code units. Typically 1. 
        time:[float] Amount of time to create (will create 10x this amount to be safe)
    
    OPTIONAL INPUTS
        isothermal:[bool] Sets the gas behavior. If false, need to set value for adiabatic index n.
        Fmax:[float] Maximum frequency in turbulence spectrum. Default is 2 hours.
        n:[float] Adiabatic index. gamma = (n+1)/n
    
    AUTHOR:
        Connor Robinson, May 17th, 2019
    '''
    #Define constants
    kb = 1.38e-23 # J/K
    mp = 1.67e-27 # kg
    
    #Assume solar metalicity
    Xm = 0.7381
    Ym = 0.2485
    Zm = 0.0134
    mu = (2*Xm + 3/4*Ym + 1/2*Zm)**(-1)
    
    #Constants
    G = 6.67e-11 # mks
    Rsun = 6.95508e8 #m
    Msun = 2e30 #kg
    
    #Convert time into code units
    if isothermal:
        gamma = 1
    else:
        gamma = (n+1)/n
    
    #Set stellar parameters
    cs = np.sqrt(gamma * kb * Tdisk / (mu * mp))
    
    #Derive maximuum and minimum frequencies
    Kmax = 2 * np.pi * (Rstar * Rdisk * Rsun)
    
    Fmin = np.log10(cs / Kmax) #Set by sound speed and circumference of disk
    Fmax = np.log10(2.0/(60 * 60)) #Set minimum timescale to 2 hours
    
    #Calculate necessary number of cells. Increase by a factor of 10 to be safe.
    N = 10 * np.ceil(2 * 10**Fmax * (time * 24 * 60 * 60)) + 1 
    
    #Generate the turbulence spectrum
    T, sig = kolmogorav(N, Fmin, Fmax)
    
    #Scale the velocity output by the sound speed and the amplitude of the variations
    if len(np.atleast_1d(amplitude)) > 1:
        
        boundary_file = []
        
        for i, a in enumerate(amplitude):
            sig_cs = sig/cs
            rms = np.std(sig_cs)
            scale_sig = sig_cs * (a/rms)
            
            T_code = convert_time(T, Rstar, Tdisk, gamma, mu = mu)
            
            #Set the density equal to the disk density. 
            rho = np.zeros(len(T)) + rho_disk
            
            #Save it
            np.savetxt(simpath+tagbase+'_kolmogorav_'+str(i+1)+'.dat', np.vstack([T_code, scale_sig, rho]).T)
            boundary_file.append(clusterpath+tagbase+'_kolmogorav_'+str(i+1)+'.dat')
        
        boundary_file = np.array(boundary_file)
        return boundary_file
        
    else:
        sig_cs = sig/cs
        rms = np.std(sig_cs)
        scale_sig = sig_cs * (amplitude/rms)
        
        T_code = convert_time(T, Rstar, Tdisk, gamma, mu = mu)
        
        #Set the density equal to the disk density. 
        rho = np.zeros(len(T)) + rho_disk
        
        #Save it
        np.savetxt(simpath+tagbase+'_kolmogorav.dat', np.vstack([T_code, scale_sig, rho]).T)
        
        boundary_file = clusterpath+tagbase+'_kolmogorav.dat'
        
        return boundary_file

def convert_time(time, Rstar, Tdisk, gamma, mu = 1):
    '''
    
    PURPOSE:
        Convert physical time (in days) to code units (sound crossing times)
    
    INPUTS:
        time:[float] Value to convert
        Rstar:[float] Stellar radius in Rsun
        Tdisk:[float] Temperature of the disk
        gamma:[float] Adiabatic index
    
    OPTIONAL INPUTS:
        mu:[float] Molecular mass. Default is 1. 
    
    INPUTS:
    
    '''
    
    #Define constants
    kb = 1.38e-23 # J/K
    mp = 1.67e-27 # kg
    G = 6.67e-11 # mks
    Rsun = 6.95508e8 #m
    
    R = Rstar * Rsun
    cs = np.sqrt(gamma * kb * Tdisk / (mu * mp))

    #Convert into code units
    time_code = cs/R * time * (60*60*24)
    
    return time_code

# def getgammaSurf(rin, GAMMA, method = 'numeric'):
#     '''
#
#     getThetaSurf
#
#     PURPOSE:
#         Get the angle at which the field line connects to the surface of the star
#
#     INPUTS:
#         rin:[float] Co-rotation radius
#         GAMMA:[float] Octupole contribution
#
#     AUTHOR:
#         Connor Robinson, October 29th, 2018
#     '''
#
#     q = 1/rin * (1 - GAMMA/(4 * rin))
#
#
#     if method == 'symbolic':
#         theta = sp.Symbol('theta')
#
#         print('Solving for theta...')
#         gamma0 = np.array(sp.solvers.solve(1/4 * GAMMA * (5 * sp.cos(theta)**2 -1) * sp.sin(theta)**2 + sp.sin(theta)**2 - q))
#         print('Finished')
#
#         #Choose the smallest positive root and convert to degrees
#         root = np.float(gamma0[gamma0 > 0][0] * 180/sp.pi)
#
#     elif method == 'numeric':
#
#         x0 = 1/(rin * (1+GAMMA))
#         root = optimize.root(gammafunc, x0, args = (q, GAMMA), method='lm')['x'][0] * 180/np.pi
#
#     return root

def gammafunc(theta, q = None, GAMMA = None):
    # return 1/4 * GAMMA * (5 * sp.cos(theta)**2 -1) * sp.sin(theta)**2 + sp.sin(theta)**2 - q
    return 1/4 * GAMMA * (5 * math.cos(theta)**2 -1) * math.sin(theta)**2 + math.sin(theta)**2 - q


def getBand(wl, flux, bandfile):
    '''
    
    connector.getBand
    
    PURPOSE:
        Convert spectra into photometry point. TRANSMISSION CURVES MUST BE IN NM
    
    INPUTS:
        wl:[array] Wavelength array IN MICRONS
        flux:[array] Flux array
        bandfile:[str] File containing the transmission curve IN NM
    
    AUTHOR:
        Connor Robinson, October 26th, 2018
    
    
    '''
    #Load in the transmission window
    trans = np.genfromtxt(bandfile, skip_header = 1)
    
    wl_t = trans[:,0]/1e3  #microns
    filt = trans[:,1]      # T/per micron
    
    filt[filt < 0] =0 
    filt = filt/np.max(filt)
    
    #Remove bad points
    good = np.isfinite(flux)
    
    # For each band for the given instrument
    # First it interpolates the model at the wavelengths for which
    # we have values of the transmissivity
    F_filt = np.interp(wl_t, wl[good], flux[good])
    
    # Now it integrates and convolves the model with the transmissivity
    # of that particular band. It uses a trapezoidal integration.
    s1 = np.trapz(F_filt*filt, x = wl_t)
    # It integrates the transmissivity to normalize the convolved flux
    s2 = np.trapz(filt, x = wl_t)
    
    intF = s1 / s2
    
    return intF # Synthetic flux at the band


def getQ(t, lc, err, pt = 0.2, tau = 1, ell = 0.3, showplot = False, plotname = None, \
         T0 = None, gauss = False, manualACF = False, GP = True, boxcar = False, getPeriod = False, getModel = False):
    '''
    
    getQ
    
    PURPOSE:
        Calculate the Q metric from Cody et al. 2014. 
        This metric measures the periodicity of a light curve.
        
        It compares the residuals after subtracting a sinusoidal signal to the raw residuals. 
        
    INPUTS:
        t:[float arr] Time -- This must be evenly spaced for the autocorrelation function!
        lc:[float arr] Light curve in flux units. 
        err:[float] Charachteristic uncertainty for all points
    
    OPTIONAL INPUTS:
        T0: If given, fix the period to this value. 
        pt: Period threshold. Sets the maximum range away from the peak of the ACF to 
            search for periods with the Lomb-Scargle periodogram
        tau:[float] "Strength" of the covariance
        ell:[float] Width of the covariance
        showplot:[bool] Turn on to show plots on screen
        plotname:[str] If specified, plot will save to this file.
        gauss:[bool] If true, use a Gaussian weighting function instead of a boxcar to select the period.
        manualAcf:[bool] If true, lets you select the desired peak of the ACF by hand.
        GP:[bool] If true, uses Gaussian processes to fit the curve. If false, uses the Supersmoother algorithm
        
    AUTHOR:
        Connor Robinon, May 14th, 2019
    '''
    
    #Get the ACF
    ################################
    
    #Resample the data onto a uniform grid
    print('Interpolating onto uniform grid')
    tu = np.linspace(t[0], t[-1], len(t))
    func = interpolate.interp1d(t,lc, kind = 'cubic')
    lcu = func(tu)
    
    print('Obtaining ACF')
    ACF = getACF(lcu)
    
    #Identify peaks in the ACF
    ################################
    
    #Choose the amount of cells to be smoothed over.
    stime = 0.5 #days
    Ncells = np.argmin(np.abs(t - (stime+t[0])))
    
    if np.mod(Ncells,2) == 0:
        Ncells = Ncells + 1
    
    #Smooth the ACF with the Savitsky-Golay filter to avoid issues with noise.
    smooth = scipy.signal.savgol_filter(ACF, Ncells, 3, mode = 'interp')
    
    maxs = signal.argrelextrema(smooth, np.greater)[0]
    mins = signal.argrelextrema(smooth, np.less)[0]
    
    #The first peak should always be after the first minimum (since the highest value in the ACF is lag = 0)
    #Make sure that I end on a minimum
    if len(mins) == len(maxs) and len(maxs) > 1:
        maxs = maxs[:-1]
    
    #Check to see if the surrounding maxima are at least 0.05 greater than the surrounding minima
    bigmax = []
    
    if manualACF:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t[:-1] - t[0], smooth, color = 'k')
        for x in maxs:plt.axvline(t[x] - t[0])
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block = True)
        fig.canvas.mpl_disconnect(cid)
        
        global coords
        
        guessACF = coords[0]
        
        bigmax = [maxs[np.argmin(np.abs(t[maxs] - t[0] - guessACF))]]
    
    else:
        for i, m in enumerate(maxs):
            if (ACF[m] > ACF[mins[i]]+0.05) and (ACF[m] > ACF[mins[i+1]]+0.05):
                bigmax.append(m)
    
    
    #Select the biggest peak from the first two peaks in bigmax
    if len(bigmax) == 0:
        print('No good peaks found')
        return np.nan
        maximum = np.nan
    if len(bigmax) == 1:
        maximum = bigmax[0]
    if len(bigmax) >= 2:
        maximum = bigmax[np.argmax(ACF[bigmax[:2]])]
    
    peak = t[maximum] - t[0]
    
    #Compute a periodogram and select peak within a given threshold of the peak from frequency from the ACF
    ################################
    #nu, po = LombScargle(t, lc).autopower()
    #T = 1/nu
    
    nu = 10**np.linspace(-2,1, 1000)
    po = LombScargle(t, lc).power(nu)
    T = 1/nu
    
    if T0 == None:
        if gauss:
            
            thresh = peak * pt
            
            weighting = 1/np.sqrt(2 * np.pi * thresh**2) * np.exp(-(T - peak)**2/(2*thresh**2))
            weighted = po * weighting
            
            T0 = T[np.argmax(weighted)]
            
        else:
            #Isolate the region near the peak from the ACF
            window = (T > peak * (1 - pt)) * (T < peak * (1+pt))
            
            #If the period is not fixed, pick peak of ACF.
            T0 = T[window][np.argmax(po[window])]
    
    tf = t % T0
    
    if GP:
        #Fold light curve at the period and smooth with Gaussian Processes
        ################################
        #Reproduce the array 3 times to simulate periodic boundary conditions
        print('Running Gaussian Processes...')
        
        med = np.median(lc)
        x = np.hstack([tf/T0, tf/T0+1, tf/T0+2])
        y = np.hstack([lc/med]*3)
        
        e = (np.ones(len(y)) * err/med)
        xs = (tf/T0+1)
        
        emu, cov = gaussP(tau,ell,x,y,e,xs,periodic = False)
        mod = emu * med
    
    elif boxcar:
        print('Running Boxcar smoothing')
        #Do a boxcar smooth with a width of 25% of the period.
        width = 0.25
        
        #Interpolate the data onto a uniform grid
        med = np.median(lc)
        x = np.hstack([tf/T0, tf/T0+1, tf/T0+2])
        y = np.hstack([lc/med]*3)
        
        emu = []
        for xi in tqdm(tf/T0):
            region = (x > xi+1-width/2) * (x < xi+1+width/2)
            emu.append(np.nanmean(y[region]))
        emu = np.array(emu)
        
        mod = emu * med
    
    else:
        
        # Use supersmoother at the period to get the periodic part of the function
        model = SuperSmoother(period = T0)
        
        e = np.ones(len(lc)) * err
        model.fit(t, lc, e)
        
        mod = model.predict(t)
    
    #Subtract smoothed curve from the raw light curve
    ################################
    resid = lc - mod
    
    #Calculate RMS for residuals and the raw light curve
    ################################
    N = len(lc)
    
    rms_resid = np.sqrt(np.sum(resid**2)/N)
    rms_raw = np.sqrt( np.sum((lc - np.mean(lc))**2)/N )
    
    #Calculate the Q metric
    ################################
    Q = (rms_resid**2 - err**2)/(rms_raw**2 - err**2)
    
    #Make a big summary plots if requested. 
    ################################
    if showplot or plotname:
        
        cmap = cm.Spectral
        colors = np.array([cmap(0.99), cmap(.9), cmap(.75), cmap(0.40), cmap(0.1), cmap(0.2)])
        
        # Set up plot
        fig, ax = plt.subplots(3,2, figsize = [9,9])
        
        fig.subplots_adjust(hspace = 0.35, top = .96, bottom = 0.1, wspace = 0.25)
        
        titlesize = 10
        
        # Raw lc
        ax[0,0].errorbar(t, lc, yerr = err, marker = 'o', color = 'k', ls = '', markersize = 2)
        ax[0,0].set_xlabel(r'Time $[d]$')
        ax[0,0].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[0,0].set_title('a) Light Curve', loc = 'right', fontsize = titlesize)
        
        # ACF
        ax[1,0].plot(t[:-1] - t[0], ACF, color = 'k')
        ax[1,0].axvline(peak, color = colors[0], ls ='-', lw = 2, alpha = 0.7)
        ax[1,0].axvline(T0, color = colors[5], ls ='-', lw = 2, alpha = 0.7)
        ax[1,0].set_xlabel(r'Lag $[d]$')
        ax[1,0].set_ylabel(r'ACF')
        ax[1,0].set_title('b) Autocorrelation Function', loc = 'right', fontsize = titlesize)
        
        # Periodogram
        ax[2,0].step(T, po, color = 'k', where = 'mid')
        ax[2,0].axvline(peak, color = colors[0], ls ='-', lw = 2, alpha = 0.7)
        ax[2,0].axvline(T0, color = colors[5], ls ='-', lw = 2, alpha = 0.7)
        ax[2,0].set_xlabel(r'Period $[d]$')
        ax[2,0].set_ylabel(r'Power $[arb.]$')
        ax[2,0].set_xscale('log')
        ax[2,0].set_xlim(left = 0.1)
        ax[2,0].set_title('c) Periodogram', loc = 'right', fontsize = titlesize)
        
        if gauss:
            ax[2,0].step(T, weighted * po[np.argmax(weighted)]/np.max(weighted), color = 'purple', where = 'mid')
        
        # Folded lc + GP solution
        ax[0,1].plot((tf/T0)[np.argsort(tf)], mod[np.argsort(tf)], color = colors[4], zorder = 2, lw = 3, alpha = 0.7)
        ax[0,1].plot(tf/T0, lc, marker = 'o', markersize = 2, color = 'k', ls = '', zorder = 1)
        ax[0,1].set_xlabel(r'Phase')
        ax[0,1].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[0,1].set_title('d) Folded Light Curve + Fit', loc = 'right', fontsize = titlesize)
        
        #lc + GP Solution
        ax[1,1].plot(t, mod, color = colors[4], zorder = 2, lw = 3, alpha = 0.7)
        ax[1,1].plot(t, lc, marker = 'o', markersize = 2, color = 'k', ls = '', zorder = 1)
        ax[1,1].set_xlabel(r'Time $[d]$')
        ax[1,1].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[1,1].set_title('e) Light Curve + Fit', loc = 'right', fontsize = titlesize)
        
        # Residuals + rms values
        ax[2,1].plot(t, resid, marker = 'o', markersize = 2, color = 'k', ls = '', zorder = 1)
        ax[2,1].axhline(rms_resid, color = colors[1], ls = '-', lw = 4, alpha = 0.7)
        ax[2,1].axhline(rms_raw, color = colors[2], ls = '-', lw = 4, alpha = 0.7)
        ax[2,1].axhline(0, color = colors[4], ls = '-', lw = 4, alpha = 0.7)
        ax[2,1].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[2,1].set_xlabel(r'Time $[d]$')
        ax[2,1].set_title('f) Residuals', loc = 'right', fontsize = titlesize)
        
        if plotname:
            plt.savefig(plotname)
        
        if showplot:
            plt.show(block = True)
        else:
            plt.close()
    
    if getPeriod:
        if getModel:
            return Q, T0, mod
        else:
            return Q, T0
    elif getModel:
        return Q, mod
    else:
        return Q

def getM(t, lc, ell, err, tau = 1, showplot = False, plotname = None, percentile = 10, GP = True, boxcar = False, span = [0.001, 0.005, 0.01]):
    '''
    
    getM
    
    PURPOSE:
        Calculate the M metric from Cody et al. 2014. 
        This metric measures the symmetry of a light curve.
    
    INPUTS:
        t:[float arr] Time
        lc:[float arr] Light curve in flux units. 
        ell:[float] Length of covariance time for smoothing
        err:[float arr] Charachteristic uncertainity for all points.
    
    OPTIONAL INPUTS:
        tau:[float] Strength of the correlations. Default is 1. 
        showplot:[bool] Show plots on screen
        plotname:[str] If defined, will save plot to this file
        percentile:[float] Set the percentile to select when computing the means. 
                           Default is 10% from Cody et al. 2014. However, Cody et al. (2017) uses 5%
    
    AUTHOR:
        Connor Robinon, May 14th, 2019
    '''
    
    if GP:
        
        #Use Gaussian Processes to smooth the light curve to find outliers
        ################################
        N = len(lc)
        
        #Normalize the light curve by the median (Fixes numerical issues + use generic tau = 1)
        med = np.median(lc)
        y = lc/med
        e = np.ones(N)*(err/med)
        
        
        print('Running Gaussian Processes...')
        emu, cov = gaussP(tau,ell,t,y,e,t)
        mod = emu * med
    
    elif boxcar:
        print('Running Boxcar smoothing')
        
        #Interpolate the data onto a uniform grid
        med = np.median(lc)
        x = t
        y = lc/med
        
        emu = []
        for xi in tqdm(x):
            region = (x > xi-ell/2) * (x < xi+ell/2)
            emu.append(np.nanmean(y[region]))
        emu = np.array(emu)
        
        mod = emu * med
    
    else:
        
        alpha = 0
        
        # Use supersmoother at the period to get the periodic part of the function
        
        #Calculate the spans -- assume the time is given in hours
        model = SuperSmoother(primary_spans = span, middle_span = span[1], final_span = span[0])
        
        e = np.ones(len(lc)) * err
        model.fit(t, lc, e)
        
        mod = model.predict(t)
    
    #Subtract the smoothed curve
    ################################
    resid = lc - mod
    
    #Remove 5 sigma outliers if any
    ################################
    rms = np.std(resid)#np.sqrt(np.sum(resid**2)/N)
    clip = ~(np.abs(resid) > rms * 5)
    
    #Isolate the top/bottom 10% of points in the clipped array
    ################################
    top = np.percentile(lc[clip], 100 - percentile)
    bot = np.percentile(lc[clip], percentile)
    
    extr = ((lc > top) + (lc < bot)) * clip
    
    #Find the mean of the top/bottom 10% points and the mean of the clipped light curve
    ################################
    d10 = np.mean(lc[extr])
    dmean = np.mean(lc[clip])
    
    #Compute the RMS
    errd = np.std(lc)
    
    # old = np.sqrt(np.sum((lc)**2/N))
    # plt.plot(t, lc, color = 'k')
    # plt.axhline(old, color = 'r')
    # plt.axhline(errd, color = 'b')
    # plt.ylim(ymin = 0)
    # plt.show()
    
#    pdb.set_trace()
    
    #Calculate M
    ################################
    
    #NOTE -- Added a minus sign here because the original paper (Cody et al. 2014) did everything with magnitudes. 
    #This follows the K2 paper, which included the minus sign. 
    M = -(d10 - dmean)/errd
    
    if showplot or plotname:
        
        cmap = cm.Spectral
        colors = np.array([cmap(0.99), cmap(.9), cmap(.75), cmap(0.40), cmap(0.1), cmap(0.2)])
        
        # Set up plot
        fig, ax = plt.subplots(2,2, figsize = [9,6])
        
        fig.subplots_adjust(hspace = 0.35, top = .96, bottom = 0.1, wspace = .25)
        
        titlesize = 10
        
        # Raw light curve with errors
        ax[0,0].errorbar(t, lc, yerr = err, marker = 'o', color = 'k', ls = '', markersize = 2)
        ax[0,0].set_xlabel(r'Time $[d]$')
        ax[0,0].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[0,0].set_title('a) Light Curve', loc = 'right', fontsize = titlesize)
        
        # Light curve + GP solution 
        ax[1,0].plot(t, lc, marker = 'o', color = 'k', ls = '', markersize = 2)
        ax[1,0].plot(t, mod, color = colors[4], alpha = 0.7, lw = 2.5)
        ax[1,0].set_xlabel(r'Time $[d]$')
        ax[1,0].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[1,0].set_title('b) Light Curve + GP solution', loc = 'right', fontsize = titlesize)
        
        # Residuals + outlier cuts
        ax[0,1].plot(t[clip], resid[clip], marker = 'o', color = 'k', ls = '', markersize = 2)
        ax[0,1].axhline(rms * 5, color = colors[2], ls = '-', lw = 3, alpha = 0.7)
        ax[0,1].axhline(rms * -5, color = colors[2], ls = '-', lw = 3, alpha = 0.7)
        ax[0,1].plot(t[~clip], resid[~clip], color = colors[2], marker = 'o', ls = '', markersize = 2)
        ax[0,1].set_xlabel(r'Time $[d]$')
        ax[0,1].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[0,1].set_title('c) Residuals', loc = 'right', fontsize = titlesize)
        
        # Light curve with isolated points + horizontal lines at means
        ax[1,1].plot(t[~extr], lc[~extr], marker = 'o', color = 'k', ls = '', markersize = 2)
        ax[1,1].plot(t[extr], lc[extr], marker = 'o', color = colors[0], ls = '', markersize = 2)
        ax[1,1].plot(t[~clip], lc[~clip], marker = 'o', color = colors[2], ls = '', markersize = 2)
        
        ax[1,1].axhline(top, color = colors[0], ls ='-', lw = 3   , alpha = 0.7)
        ax[1,1].axhline(bot, color = colors[0], ls = '-', lw = 3  , alpha = 0.7)
        ax[1,1].axhline(d10, color = colors[1], ls = '-', lw = 3  , alpha = 0.7)
        ax[1,1].axhline(dmean, color = colors[4], ls = '-', lw = 3, alpha = 0.7)
        
        ax[1,1].set_xlabel(r'Time $[d]$')
        ax[1,1].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[1,1].set_title('d) Top & bottom '+str(percentile)+'%', loc = 'right', fontsize = titlesize)
#        ax[1,1].set_ylim(ymin = 0)
        
        
        if plotname:
            plt.savefig(plotname)
        
        if showplot:
            plt.show(block = True)
        else:
            plt.close()
    
    return M

def onclick(event):
    '''
    onclick
    PURPOSE:
        Small helper function to get coordinates on click
    
    AUTHOR:
        Connor Robinson, July 1st, 2019
    '''
    ix, iy = event.xdata, event.ydata
    
    global coords
    coords = [ix, iy]
    plt.close()
    return

def getACF(d):
    '''
    
    getACF
    
    PURPOSE:
        Calculate the ACF given a light curve following Cody et al. 2014
    
    INPUTS:
        d:[float arr] Light curve. Temporal spacing is assumed to be uniform. 
    
    AUTHOR:
        Connor Robinson, May 14th, 2019
    '''
    
    dmean = np.mean(d)
    
    acf = []
    
    N = len(d)
    
    iall = np.arange(0, N-1)
    
    acf = []
    
    lags = np.arange(0, N-1)
    
    for lag in lags:
        #for i, di in enumerate(d):
        i = np.arange(0, N-lag - 1)
        ilag = i + lag
        
        #This might be slightly off, since I am dividing by a larger number of points as the lag becomes larger. 
        acf.append(np.sum((d[i] - dmean) * (d[ilag] - dmean)/np.sum( (d[iall] - dmean)**2)))
    
    acf = np.array(acf)
    
    return acf
    

def Kfunc(tau, ell, x, xi):
    '''
    
    Kfunc()
    
    PURPOSE:
        Get the covariance for a point given strength and length of the covariance and the seperation of the points.
        Assumed to be a Gaussian.
    
    INPUTS:
        tau:[float] Strength of the covariance
        ell:[float]  Lenght of the covariance
        x:[array/float] x values, either single value or from observations
        xi:[float] Single value to get covariance compared to x
    
    OUTPUTS:
        Array with covariance given separation
    
    AUTHOR:
        Connor Robinson, April 18th, 2018
    '''
    return tau * np.exp(-(x - xi)**2/ell**2)

def Kfunc_periodic(tau, ell, x, xi):
    '''
    
    Kfunc_periodic()
    
    PURPOSE:
        Get the covariance for a point given strength and length of the covariance and the seperation of the points.
        Assumed to be periodic
    
    INPUTS:
        tau:[float] Strength of the covariance
        ell:[float]  Lenght of the covariance
        x:[array/float] x values, either single value or from observations
        xi:[float] Single value to get covariance compared to x
    
    OUTPUTS:
        Array with covariance given separation
    
    AUTHOR:
        Connor Robinson, May 28th, 2019
    '''
    return tau * np.exp(-1/2 * np.sin((x-xi)/2)**2/ell**2)

def gaussP(tau,ell,x,y,e,xs, periodic = False):
    '''
    GP()
    
    PURPOSE:
        Produces a fit for 1D data over a specified range using gaussian processes 
        Only works with data with non-correlated uncertainties
    
    INPUTS:
        tau:[float] Strength of the covariance
        ell:[float] Length of the covariance
        
        x:[array] 1D array of x values from measurement 
        y:[array] 1D array of y values from measurement
        e:[array] 1D array of y uncertainties from measurement
        
        xs:[array] 1d array of x values to solve for the fit
        
    OUTPUTS:
        Emu, cov: Mean and variance of the fit
    
    AUTHOR:
        Connor Robinson, April 18th, 2018
    '''
    
    #Make error matrix
    E = np.diag(e)
    
    #Create K array
    K = []
    for xi in x:
        K.append(Kfunc(tau, ell, x, xi))
    K = np.array(K)
    
    Emu = []
    cov = []
    
    invKE = np.linalg.inv(K+E)
    
    
    if periodic:
        #Create rows based on guess
        for g in tqdm(xs):
            
            #Create the subarrays for the block matrix
            Kpred = Kfunc_periodic(tau, ell, x, g)
            cov.append(tau - np.matmul(Kpred, np.matmul(invKE, Kpred.T)))
            Emu.append(np.matmul(Kpred, np.matmul(invKE, y)))
    
    else:
        #Create rows based on guess
        for g in tqdm(xs):
            
            #Create the subarrays for the block matrix
            Kpred = Kfunc(tau, ell, x, g)
            cov.append(tau - np.matmul(Kpred, np.matmul(invKE, Kpred.T)))
            Emu.append(np.matmul(Kpred, np.matmul(invKE, y)))
        
    cov = np.array(cov)
    Emu = np.array(Emu)
    
    return Emu, cov


######################################################
#                                                    #
# CODE PAST HERE IS USED FOR BUILDING HOT SPOTS      #
#                                                    #
######################################################

def getArea(radius_deg, beta_deg):
    '''
    
    getArea
    
    PURPOSE:
        Calculates projected hot spot size given angular width and angular impact parameter
    
    INPUTS:
        radius_deg:[float] angular radius in degrees
        beta_deg:[float] angular impact parameter in degrees
    
    NOTES:
        Much of this is based on Carlos (2018), Stratified Sampling of Projected Spherical Caps
    
    '''
    
    alpha = radius_deg * np.pi/180.0
    beta = beta_deg * np.pi/180.0
#    beta = (90.0 - beta_deg) * np.pi/180.0
    
    #Define basic quantities
    ax = np.sin(alpha) * np.cos(beta)
    ay = np.sin(alpha)
    xe = np.sin(beta) * np.cos(alpha)
    
    #ax = np.sin(alpha) * np.sin(beta)
    #ay = np.sin(alpha)
    #xe = np.cos(alpha) * np.cos(beta)
    
    #If the spot is not visible:
    if beta_deg - radius_deg >= 90:
        area = 0
    
    #If spot fully visible (ellipse only):
    elif (beta_deg + radius_deg < 90):
        area = np.pi * ax * ay
    
    #If spot is partially hidden on other side of star (ellipse + lune):
    elif (beta_deg + radius_deg >= 90) and (beta_deg <= 90):
        #yl = np.sqrt(1 - np.cos(alpha)**2/np.cos(beta)**2)
        yl = np.sqrt(1 - np.cos(alpha)**2/np.sin(beta)**2)
        # ( Outer circle ) - ( inner ellipse ) + (area of ellipse)
        area = eI(yl) - 2*yl*xe - ax*ay*eI(yl/ay) +  np.pi*ax*ay 
        
    #If spot is mostly hidden on other side of the star (lune only):
    elif (beta_deg > 90) and (beta_deg < 90 + radius_deg):
        #yl = np.sqrt(1 - np.cos(alpha)**2/np.cos(beta)**2)
        yl = np.sqrt(1 - np.cos(alpha)**2/np.sin(beta)**2)
        
        # ( Outer circle ) - ( inner ellipse ) NOTE THE SIGN CHANGE (seeing other side of ellipse)
        area = eI(yl) - 2*yl*xe + ax*ay*eI(yl/ay) 
    
    if area < 0:
        area = 0
        
    
    return area/np.pi #area of unit circle is pi

def eI(u, w = 1):
    '''
    eI
    
    PURPOSE:
        Ancillary function for I (eqn. 17)
    
    AUTHOR:
        Connor Robinson, October 22nd, 2018
    '''
    
    return (w * u * np.sqrt(1 - u**2) + np.arcsin(u))
    

def getBeta(phi_deg, gamma_deg, i_deg):
    '''
    getAB
    
    PURPOSE:
        Converts from rotational phase and spot latitude into alpha and beta for getArea
    
    INPUTS:
        phi_deg:[float] rotational phase in degrees. Defined such that 0 is face on.
        gamma_deg:[float] angle between rotation axis and center of spot
        i_deg:[float] inclination in degrees
    '''
    
    if gamma_deg > 90 or i_deg > 90:
        print('Gamma and inclination must be < 90 degrees. Returning NaN')
        return np.nan
    
    phi = phi_deg * np.pi/180
    gamma = gamma_deg * np.pi/180#(90 - gamma_deg) * np.pi/180
    i = i_deg * np.pi/180
    
    x = -np.sin(gamma) * np.cos(i) * np.cos(phi) + np.cos(gamma)*np.sin(i)
    y = np.sin(gamma) * np.sin(phi)
    z = np.sin(gamma)*np.sin(i)*np.cos(phi) + np.cos(i) * np.cos(gamma)
    
    
    b = np.sqrt(x**2 + y**2)
    
    if z > 0:
        beta = np.arcsin(b) * 180/np.pi
    
    else:
        #Handle the case where beta is greater than 90 degrees
        beta = 180 - np.arcsin(b)* 180/np.pi 
    return beta
    


######################################################
#                                                    #
# CODE PAST HERE IS USED FOR CONSTRUCTION OF PATCHES #
#                                                    #
######################################################

def pgi_to_xyz(phi, gamma, i):
    '''
    pgi_to_xyz
    
    PURPOSE:
        Transform phase, gamma, and inclination into cartesian coordinates
    
    INPUTS:
        phi:[float] Phase
        gamma:[float] angular distance from the pole
        i:[float] Inclination (90 is edge-on)
    
    AUTHOR:
        Connor Robinson, Sept. 8, 2020
    
    '''
    
    x = np.sin(gamma) * np.sin(phi)
    y = -np.sin(gamma) * np.cos(i) * np.cos(phi) + np.cos(gamma) * np.sin(i)
    z = np.sin(gamma) * np.sin(i) * np.cos(phi) + np.cos(i) * np.cos(gamma)
    
    return np.array([x,y,z]).T

def xyz(nw, g1, g2, width, i, phase):
    '''
    
    xyz
    
    PURPOSE:
        Define coordinates of strips at equal latitudes.
    
    INPUTS:
        nw:[int] Number of cells in the strip
        g1:[float] Angle between top of strip and rotational pole
        g2:[float] Angle between bottom of strip an rotational pole. 
        width:[float] Angular width of strip
        i:[float] Inclination
        phase:[float] Phase of the strip. 
    
    AUTHOR:
        Connor Robinson, Sept. 8th, 2020
    '''
    
    #Divide the phases equally
    phi_1 = np.linspace(phase, phase+width, nw+1)[:-1]
    phi_2 = np.linspace(phase, phase+width, nw+1)[1:]
    
    # Define coordinates for each of the four vertices.
    p1 = {'phi':phi_1, 'gamma':np.ones(nw)*g1}
    p2 = {'phi':phi_2, 'gamma':np.ones(nw)*g1}
    p3 = {'phi':phi_1, 'gamma':np.ones(nw)*g2}
    p4 = {'phi':phi_2, 'gamma':np.ones(nw)*g2}
    
    # Calculate cartesian coordinates for each
    c1 = pgi_to_xyz(p1['phi'], p1['gamma'], i)
    c2 = pgi_to_xyz(p2['phi'], p2['gamma'], i)
    c3 = pgi_to_xyz(p3['phi'], p3['gamma'], i)
    c4 = pgi_to_xyz(p4['phi'], p4['gamma'], i)
    
    return c1, c2, c3, c4


def strip_area(nw, g1, g2, width, i, phase):
    '''
    iarea
    
    PURPOSE:
        Returns the projected area of the spot (for a single phase)
    
    INPUTS:
        nw:[int] Number of cells along stripe of constant latitude
        g1:[float] Magnetic footprint closest to the pole
        g2:[float] Magnetic footprint closest to the equator
        width:[float] Angular width of the patch
        i:[float] Inclination
        phase:[float] Rotational phase
    
    AUTHOR:
        Connor Robinson, Sept. 8th, 2020
    '''
    
    d2r = np.pi/180
    
    c1,c2,c3,c4 = xyz(nw, g1*d2r, g2*d2r, width*d2r, i*d2r, phase*d2r)
    
    #Define vectors to help calculate the surface area
    Q = c2 - c1
    R = c3 - c1
    S = c2 - c4
    T = c3 - c4

    #Calculate the area of the two triangles that make up each cell.
    n1 = np.cross(R,Q)/2
    n2 = np.cross(S,T)/2
    
    proj1 = np.dot(n1, [0,0,1])
    proj2 = np.dot(n2, [0,0,1])
    
    #Remove regions behind the star
    proj1[proj1 < 0] = 0
    proj2[proj2 < 0] = 0
    
    f = np.sum(proj1 + proj2)/(np.pi)
    
    return f

def get_theta(r, rdisk, GAMMA, Z = 0.05):
    '''
    get_theta
    
    Calculates theta (polar coordinates) along a field line.
    
    NOTE: 
    
    INPUTS:
        r:[float] Radius in stellar radii (r = 1 -> stellar magnetic footprint)
        rdisk:[float] Location of magnetic footprint at disk in stellar radii
        GAMMA:[float] Octupole/dipole ratio
        
    OPTIONAL INPUTS:
        Z = 0.05: Height of last cell in simulation in stellar radii (at Rdisk)
    
    AUTHOR:
        Connor Robinson, Sept 14th, 2020
    '''
    theta_o = np.pi/2 - np.arctan(Z/rdisk)
    p,q = get_pq(rdisk, theta_o, GAMMA)
    
    
    if GAMMA == 0:
        GAMMA = 0.0001
    
    theta = np.arcsin(np.sqrt( 2.0/(5.0*GAMMA) * ((r**2.0 + GAMMA) - ((r**2.0 + GAMMA)**2.0 - 5.0*GAMMA*q*r**3.0)**(1.0/2.0))))
    
    return theta

def get_pq(r,theta, GAMMA):
    '''
    
    get_qp
    
    PURPOSE:
        r:[float] Radius in stellar radii
        theta:[float] Angle from rotational pole.
        GAMMA:[float] Ratio of octupole to dipole component.
    
    '''
    
    p = (-1.0/4.0) * (r**(-4.0)) * GAMMA * (5.0*np.cos(theta)**(2.0) - 3.0) * np.cos(theta) - (r**(-2.0)) * np.cos(theta)
    q = (1.0/4.0) * (r**(-3.0)) * GAMMA * (5.0*np.cos(theta)**(2.0) - 1.0) * np.sin(theta)**(2.0) + (r**(-1.0)) * np.sin(theta)**(2.0)
    
    return p, q

######################################################
#                                                    #
# CODE PAST HERE IS USED FOR CONSTRUCTION OF TRIALS  #
#                                                    #
######################################################

class trial:
    '''
    
    connector.trial(jobs, outname, path, paramfile, wtts, nzeros = 4)
    
    PURPOSE:
        Create an object for storing information + simulations
    
    AUTHOR:
        Connor Robinson, February 15h, 2019
    '''
    def __init__(self, jobs, modelname, modelpath, tablefile, paramfile, nzeros = 4, gamma = None):
        '''
        
        trial.__init__()
        
        PURPOSE:
            Initialize array and fill it with the model + information from the table.
        
        INPUTS:
            jobs:[array] Job numbers for each frame
            modelname:[str] Output name
            modelpath:[str] Output location
            tablefile:[str] Location of the input parameter file with simulation/shock parameters
            paramfile:[str] Location of the accretion shock parameter file
        
        OPTIONAL INPUTS:
            nzeros:[int] Zero padding
            gamma:[float] Angle between spot and rotation axis. If previously calculated, can enter it here to save time.
        
        AUTHOR:
            Connor Robinson, February 15h, 2019
        '''
        
        #Define constants (mks)
        Rsun = 6.957e8
        G = 6.67e-11
        Msun = 2e30
        
        #Add all the parameters from the table into the object
        table = ascii.read(tablefile)
        
        #Parameters from the simulation
        self.rdisk = table['rdisk'][0]
        self.GAMMA = table['GAMMA'][0]
        self.n = table['n'][0]
        self.timescale = table['timescale'][0]
        self.amplitude = table['amplitude'][0]
        self.delay = table['delay'][0]
        self.Mstar = table['Mstar'][0]/Msun
        self.Rstar = table['R_sol'][0]
        self.Tdisk = table['Tdisk'][0]
        self.Ncells = table['Ncells'][0]
        self.Nrun = table['Nrun'][0]
        self.Ndump = table['Ndump'][0]
        self.index = table['index'][0]
        self.rho_star = table['rho_star'][0]
        self.mdot = table['mdot'][0]
        self.isothermal = table['isothermal'][0]
        self.u_i = table['u_i'][0]
        self.cs = table['cs'][0]
        self.rho_disk = table['rho_disk'][0]
        self.Z = table['Z'][0]
        self.in_type = table['in_type'][0]
        self.out_type = table['out_type'][0]
        self.disk_variability = table['disk_variability'][0]
        
        #Parameters from shock models
        self.name = table['model'][0]
        self.dist = table['dist'][0]
        self.Teff = table['teff'][0]
        self.n0 = table['n0'][0]
        self.tstart = table['tstart'][0]
        self.tend = table['tend'][0]
        self.cadence = table['cadence'][0]
        
        #Load in each model and combine them into a single object
        flux = []
        
        print('Loading models into object')
        #Patch issue with wl for multiprocessing.
        wl = 0
        for j in tqdm(jobs):
            
            job = str(j).zfill(nzeros)
            
            mod = fits.open(modelpath+modelname+'_'+job+'.fits')
            
            #Load in each of the objects
            Fpre_model_wave = mod[0].data[mod[0].header['PREAXIS']]
            Fhp_model_wave = mod[0].data[mod[0].header['HEATAXIS']]
            
            model = Fpre_model_wave + Fhp_model_wave
            wl = mod[0].data[mod[0].header['WLAXIS']]
            
            flux.append(model)
        
        #Add the models to the object
        self.model = np.array(flux)
        self.wl = np.array(wl)
        
        #Get the time, velocity, density and kinetic energy flux from the accretion shock model parameter file
        shocktable = ascii.read(paramfile)
        
        time_arr = []
        vel_arr = []
        rho_arr = []
        F_arr = []
        
        for job in jobs:
            time_arr.append(shocktable['time'][shocktable['jobnum'] == job][0])
            vel_arr.append(shocktable['VELOCITY'][shocktable['jobnum'] == job][0])
            rho_arr.append(shocktable['RHO'][shocktable['jobnum'] == job][0])
            F_arr.append(shocktable['BIGF'][shocktable['jobnum'] == job][0])
        
        self.time = np.array(time_arr)
        self.vel = np.array(vel_arr)
        self.rho = np.array(rho_arr)
        self.F = np.array(F_arr)
        
        #Get the angle between the rotation axis and the hot spot based on the magnetic field.
        if gamma == None:
            gamma = get_theta(1, self.rdisk, self.GAMMA, Z = self.Z) * 180/np.pi
            self.gamma = gamma
        else:
            self.gamma = gamma
        
        #Calculate the period of the star in days
        corot = self.rdisk * self.Rstar * Rsun
        self.period = np.sqrt((4*np.pi**2)/(G * self.Mstar * Msun) * corot**3) * 1/(24 * 60 * 60)
        
        #Add an empty nested dictionary to hold different inclinations/spot sizes
        self.spot = {}
        
        #Add an empty dictionary to store raw light curves (i.e., before rotational modulation)
        self.lc = {}
        
        #Add an empty dictionary to store wtts flux values
        self.wttsflux = {}
        
    
    def addStrip(self, inc, ftrunc, width, phi0, name, nw = 100, nstrip = 5):
        '''
        addStrip
        
        PURPOSE:
            Generate a new strip of material and finds the projected area as a function of rotational phase.
            Adds this area to the self.spot dictionary. 
        
        INPUTS:
            i:[float] Inclination
            ftrunc:[float] Truncation radius, written as a fraction of rdisk
            width:[float] Angular width of strip
            phi0:[float] Phase offset of the strip
            name:[str] Name of the strip. 
            
        OPTIONAL INPUTS:
            nw:[int] Number of cells in each strip. Default is 100
            nstrip:[int] Number of strips. Default is 5. (increase with larger differences between g1 and g2)
        
        AUTHOR:
            Connor Robinson, Sept 8th, 2020
        '''
        
        #Calculate phase for each timestep
        t0 = self.time[0]
        phi = ( (phi0 + (self.time - t0)/self.period) % 1) * 360
        
        #Calculate the upper and lower values of gamma
        g1 = get_theta(1, self.rdisk, self.GAMMA, Z = 0.05) * 180/np.pi
        g2 = get_theta(1, ftrunc*self.rdisk, self.GAMMA, Z = 0.05) * 180/np.pi
        
        #Break area into multiple strips
        A = []
        gs1 = np.linspace(g1, g2, nstrip+1)[:-1]
        gs2 = np.linspace(g1, g2, nstrip+1)[1:]
        
        for p in phi:
            As = []
            for ns in np.arange(nstrip):
                Asi = strip_area(nw, gs1[ns], gs2[ns], width, inc, p)
                
                As.append(Asi)
            A.append(np.sum(As))
        
        A = np.array(A)
        
#        A = np.array([strip_area(nw, g1, g2, width, inc, p) for p in phi])
        
        #Add the results to the object
        self.spot[name] = {'A':A, 'Nwidth':nw, 'gamma1':g1, 'gamma2':g2, 'i':inc, 'phi0':phi0, 'phi':phi}
    
    
    def addSpot(self, inc, alpha, phi0, name, gamma = None):
        '''
        addSpot
        
        PURPOSE:
            Use the hotspot code to generate the spot area as a function of rotational phase.
            Adds this area to the self.spot dictionary
        
        INPUTS:
            inc:[float] Inclination (in degress)
            alpha:[float] Spot size (in degrees)
            phi0:[float] Fractional phase offset (e.g, phi0 = 0.5 -> 180 degrees)
            name:[str] Name to add 
        
        OPTIONAL INPUTS:
            gamma:[float] Angle between rotation axis and spot center. If 'None', default is set by spot geometry
        
        OUTPUTS:
            Adds an entry into the nested dictionary self.spots with the following tags:
            alpha: Spot size in degrees
            inc: Inclination in degrees
            A: Area as a function of phase
            phi0: Fractional phase offset. 
            phi: Fractional phase (angular phase/(2*pi))
        
        AUTHOR:
            Connor Robinson, February 15th, 2019
        '''
        
        #Set gamma to that set by magnetic field if not specified. 
        if gamma == None:
            gamma = self.gamma
        
        #Calculate phase for each timestep
        t0 = self.time[0]
        phi = ( (phi0 + (self.time - t0)/self.period) % 1) * 360
        
        #Calculate the angular impact parameter of the spot center as a function of phase
        beta = np.array([getBeta(p, gamma, inc) for p in phi])
        
        #Calculate the area of the spot as as function of phase
        A = np.array([getArea(alpha,b) for b in beta])
        
        #Add the results to the object
        self.spot[name] = {'alpha':alpha, 'inc':inc, 'A':A, 'phi0':phi0, 'phi':phi}
    
    def diffSpot(self, big, small, name):
        '''
        Create a concentric spot by subtracting the center of a bigger spot by a smaller spot
        
        INPUTS:
            big:[str] Name of the bigger spot
            small:[str] Name of the smaller spot
            name:[str] Name of the new spot
        '''
        
        #Check to see if big spot > small spot
        bigalpha = self.spot[big]['alpha']
        smallalpha = self.spot[small]['alpha']
        
        if bigalpha < smallalpha:
            raise ValueError('The big spot must be larger than the small spot.')
        
        #Check to see if phases match exactly -- This possibly could be relaxed at some point as long as small spot is entirely within big spot?
        elif (self.spot[big]['phi'] != self.spot[small]['phi']):
            raise ValueError('Phases for spots much match exactly')
        
        else:
            A = self.spot[big]['A'] - self.spot[small]['A']
        
            self.spots[name] = {'alpha_big':self.spot[big]['alpha'], 'alpha_small':self.spot[big]['alpha'], \
                                'phi0':self.spot[big]['phi0'], 'phi':self.spot[big]['phi'], 'inc':self.spot[big]['inc']}
    
    def spec2phot(self, band, wttsflux, Rwtts, Dwtts, bandpath = '/Users/connor/Dropbox/Research/burst/resources/Filters/'):
        '''
        PURPOSE:
            Converts the model spectrum into photometry values. 
            This will replace the original construction of getting the spectrum of the spot first. 
            Also will NOT use the WTTS spectrum, since those templates often lack the shortest wavelength information. 
        
        INPUTS:
            band:[str] Filter bandpass to measure photometry
            wttsflux:[float] Unscaled flux of the wtts in erg s^-1 cm^-2 (e.g., still at the distance/radius of the WTTS)
            rwtts:[float] Radius of the wtts in Rsun
            dwtts:[float] Distance to the wtts in pc
            
        OPTIONAL INPUTS:
            bandpath:[str] Location of the filter bandpasses
        
        AUTHOR: 
            Connor Robinson, December 31st, 2019
        '''
        
        #Set up filter dictionary
        filters = {'tess':bandpath + 'tess-response-function-v1.0.csv',\
                 'kepler':bandpath + 'kepler_response_hires1.txt',\
                      'U':bandpath + 'Bessel_U-1.txt',\
                      'B':bandpath + 'Bessel_B-1.txt',\
                      'V':bandpath + 'Bessel_V-1.txt',\
                      'R':bandpath + 'Bessel_R-1.txt',\
                      'I':bandpath + 'Bessel_I-1.txt',\
                      'ESPEX_NUV':bandpath + 'synthetic_ESPEX_NUV.dat',\
                      'ESPEX_O':bandpath + 'synthetic_ESPEX_O.dat',\
                      'ESPEX_NIR':bandpath + 'synthetic_ESPEX_NIR.dat'}
        
        if band not in filters:
            print('Error in spec2phot: Filter "'+band+'" not currently implemented. Returning...')
            return
        
        if band == 'kepler' or band == 'tess':
            print('Warning -- Currently cannot find absolute ZP for '+band+' for WTTS')
        
        #Calculate the photometry for the model
        bandfile = filters[band]
        
        phot = []
        for i, t in tqdm(enumerate(self.time)):
            wave_mic = self.wl/1e4
            phot.append(getBand(wave_mic, self.model[i,:], bandfile))
        
        phot = np.array(phot)
        
        self.lc[band] = phot
        
        #Scale the WTTS photometry point
        factor = (self.Rstar/Rwtts)**2*(Dwtts/self.dist)**2
        wtts = wttsflux * factor
        
        self.wttsflux[band] = wtts
        
        return
    
    def makeLC(self, band, spot):
        '''
        PURPOSE:
            Create a light curve using the previously calculated photometry
        
        INPUTS:
            band:[str] Photometric band
            spot:[str] Name of the spot
            
        
        AUTHOR:
            Connor Robinson, Dec. 31st, 2019
        '''
        
        if spot not in self.spot:
            print('Error in makeLC: Spot "'+spot+'" not found in object. Returning...')
            return 
        
        if band not in self.lc:
            print('Error in makeLC: Band "'+band+'" not found in object. Run spec2phot first. Returning...')
            return
        
        A = self.spot[spot]['A']
        
        #Calculate the photometry
        self.spot[spot][band] = self.lc[band] * A + self.wttsflux[band] * (1 - A)
        
        return self
        
    def addPhot(self, wttsname, wtts_wl, wtts_flux, Rwtts, Dwtts):
        '''
        trial.addPhot()
        
        PURPOSE:
            Add a template photosphere to the object. 
            Note: This is no longer used for obtaining flux values, and is more for display purposes.
        
        INPUTS:
            wtts_wl:[array] Wavelengths for the wtts IN MICRONS
            wtts_flux:[array] Flux info for the wtts in ergs s^-1 cm^-2
            rwtts:[float] Radius of the wtts in Rsun
            dwtts:[float] Distance to the wtts in pc
        
        AUTHOR:
            Connor Robinson, February 15th, 2019
        '''
        
        #Scale the wtts to the correct size/distance
        factor = (self.Rstar/Rwtts)**2 * (Dwtts/self.dist)**2
        
        wttsfunc = interpolate.interp1d(wtts_wl*1e4, wtts_flux * factor, bounds_error = False, fill_value = np.nan)
        self.wtts = wttsfunc(self.wl)
        
        #Add the name, radius and distance of the wtts
        self.wttsname = wttsname
        self.Rwtts = Rwtts
        self.Dwtts = Dwtts
    
    def makeSpec(self, spot):
        '''
        PURPOSE:
            Extracts spectra from a simulation + set of accretion shock models given a spot
        
        INPUTS:
            spot:[str] Name of the spot to build the spectra for
        
        OPTIONAL INPUTS:
            wttstag:[str] Tag associated with spectra in WTTS edge object. Default is 'HST'
        
        OUTPUTS:
            2D array containing spectra as a function of time
        
        AUTHOR:
            Connor Robinson, February 15th, 2019
        
        '''
        
        shockflux = []
        photflux = []
        
        for i, t in enumerate(self.time):
            
            A = self.spot[spot]['A'][i]
            
            shockflux.append(self.model[i, :] * A)
            photflux.append(self.wtts * (1 - A))
            
        shockflux = np.array(shockflux)
        photflux = np.array(photflux)
        
        self.spot[spot]['shock'] = shockflux
        self.spot[spot]['total'] = shockflux + photflux
    
    def plotSim(self, spot, band):
        '''
        
        PURPOSE:
            Plot the results of the simulation at the shock
        
        INPUTS:
            spot:[str] Name of the spot
            band:[str] Name of the photometric bandpass.
        
        '''
        
        fig, ax = plt.subplots(2, 3, figsize = [14,9])
        fig.subplots_adjust(wspace = 0.35)
        ax = ax.flatten()
        
        Rsun = 6.96e10 #cm
        Msun = 2e33 #g
        
        mdot = 8 * np.pi * (self.Rstar*Rsun/self.vel)**2 * (self.spot[spot]['A'] * self.F) * (365 * 24 * 60 * 60)/Msun
        
        med = np.median(self.spot[spot][band])
        
        ax[0].plot(self.time, self.vel/1e5, color = 'k')
        ax[1].plot(self.time, self.rho, color = 'k')
        ax[2].plot(self.time, self.F, color = 'k')
        ax[3].plot(self.time, self.spot[spot]['A'], color = 'k')
        ax[4].plot(self.time, self.spot[spot][band], color = 'k')
        ax[5].plot(self.time, mdot, color = 'k')
        
        ax[0].set_ylabel(r'$u \, [km \, s^{-1}]$')
        ax[1].set_ylabel(r'$\rho \, [g \, cm^{-3}]$')
        ax[2].set_ylabel(r'$F \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[3].set_ylabel(r'$A$')
        ax[4].set_ylabel(r'$\lambda F_\lambda \, [erg \, s^{-1} \, cm^{-2}]$')
        ax[5].set_ylabel(r'$\dot{M} \, [M_\odot \, yr^{-1}]$')
        
        ax[0].set_xlabel(r'Time $[d]$')
        ax[1].set_xlabel(r'Time $[d]$')
        ax[2].set_xlabel(r'Time $[d]$')
        ax[3].set_xlabel(r'Time $[d]$')
        ax[4].set_xlabel(r'Time $[d]$')
        ax[5].set_xlabel(r'Time $[d]$')
        
        plt.show()
        
        return
    
    def grabQ(self, spot, band, fracerr, pt = 0.2, tau = 1, ell = 0.3, showplot = True, plotname = None,
              T0 = None, gauss = False, manualACF = False, GP = True, boxcar = False):
        '''
        
        PURPOSE:
            Calculates Q for a given spot
        
        INPUTS:
            spot:[str] Name of the spot
            band:[str] Name of the photometric band
            fracerr:[str] Fractional uncertainty compared to the median
        
        OPTIONAL INPUTS:
            See getQ
        
        AUTHOR:
            Connor Robinson, Sept 24, 2020
        
        '''
        
        raw = self.spot[spot][band]
        lc = raw + np.random.normal(scale = fracerr * np.median(raw), size = len(raw))
        err = fracerr * np.median(lc)
        
        Q = getQ(self.t, lc, err, pt = pt, tau = tau, ell = ell, showplot = showplot, plotname = plotname, \
         T0 = T0, gauss = gauss, manualACF = manualACF, GP = GP, boxcar = boxcar, getPeriod = False, getModel = False)
        
        self.spot[spot]['Q'] = Q
        
        return
        
    
    def grabM(self, spot, band, fracerr, ell = 4.0/24.0, tau = 1, showplot = True, plotname = None, percentile = 5, GP = True, boxcar = False, span = [0.001, 0.005, 0.01]):
        '''
        
        PURPOSE:
            Calculates M for a given spot
        
        INPUTS:
            spot:[str] Name of the spot
            band:[str] Name of the photometric band
            fracerr:[str] Fractional uncertainty compared to the median
        
        OPTIONAL INPUTS:
            See getM
        
        AUTHOR:
            Connor Robinson, Sept 24, 2020
        
        '''
        raw = self.spot[spot][band]
        
        lc = raw + np.random.normal(scale = fracerr * np.median(raw), size = len(raw))
        err = fracerr * np.median(lc)
        
        M = getM(self.t, lc, ell, err, tau = 1, showplot = showplot, plotname = plotname, \
        percentile = percentile, GP = True, boxcar = boxcar, span = [0.001, 0.005, 0.01])
        
        self.spot[spot]['M'] = M
        
        return
    
    
    
########################################
#                                      #
#             DEFUNCT CODE             #
#                                      #
########################################



# def extract_spectra(inc, alpha, jobs, modelpath, modelname, wtts, paramfile, wttstag = 'HST'):
#     '''
#     connector.extract_spectra()
#
#     PURPOSE:
#         Extracts spectra from a simulation given inclination and spot size
#
#     INPUTS:
#         inc:[float] Inclination
#         alpha:[float] Spot size
#         jobs:[array] List of job numbers
#         modelpath:[str] Path to models
#         modelname:[str] Name of models
#         wtts:[edge object] Contains WTTS fluxes and wavelengths
#         paramfile:[str] Location of the table containing the parameters for the model
#
#     OPTIONAL INPUTS:
#         wttstag:[str] Tag associated with spectra in WTTS edge object. Default is 'HST'
#
#     OUTPUTS:
#         2D array containing spectra as a function of time
#
#     AUTHOR:
#         Connor Robinson, February 15th, 2019
#
#     '''

    #DEFUNCT CODE
        #

    #
    #
    # def makePhot(self, spot, band, basepath = '/Users/connor/Dropbox/Research/burst/resources/Filters/'):
    #     '''
    #
    #     PURPOSE:
    #         Extract the photometry for a given band for a spot
    #
    #     INPUTS:
    #         spot:[str] Name of the spot
    #         band:[str] Photometry band. Options are:
    #             'tess': TESS bandpass
    #             'kepler': Kepler Bandpass
    #             'U': Johnson-Cousins U
    #             'B': Johnson-Cousins B
    #             'V': Johnson-Cousins V
    #             'R': Johnson-Cousins R
    #             'I': Johnson-Cousins I
    #
    #
    #     AUTHOR:
    #         Connor Robinson, February 15th, 2019
    #     '''
    #     #Set up filter dictionary
    #     filters = {'tess':basepath + 'tess-response-function-v1.0.csv',\
    #              'kepler':basepath + 'kepler_response_hires1.txt',\
    #                   'U':basepath + 'Bessel_U-1.txt',\
    #                   'B':basepath + 'Bessel_B-1.txt',\
    #                   'V':basepath + 'Bessel_V-1.txt',\
    #                   'R':basepath + 'Bessel_R-1.txt',\
    #                   'I':basepath + 'Bessel_I-1.txt'}
    #
    #     bandfile = filters[band]
    #
    #     phot = []
    #
    #     for i, t in enumerate(self.time):
    #
    #         wave_mic = self.wl/1e4
    #         phot.append(getBand(wave_mic, self.spot[spot]['total'][i,:], bandfile))
    #
    #     phot = np.array(phot)
    #
    #     self.spot[spot][band+'phot'] = phot
    #

#

def compiler(jobs, outname, path, paramfile, nzeros = 4):
    '''
    connector.compiler
    PURPOSE:
        Takes all of the collated fits files and combines them into a single EDGE object.
        Returns the object and a list of jobnum vs time.
    INPUTS:
        jobs:[array] Job numbers for each frame
        outname:[str] Output name
        path:[str] Output location
        paramfile:[str] Location of the input parameter file
    OPTIONAL INPUTS:
        nzeros:[int] Amount of zero padding
    AUTHOR:
        Connor Robinson, October 26th, 2018
    '''
    
    import EDGE as edge
    
    obj = edge.TTS_Obs(outname)

    #Now load in each model and combine them into a single object
    for j in jobs:

        job = str(j).zfill(nzeros)

        mod = fits.open(path+outname+'_'+job+'.fits')

        #Load in each of the objects
        Fpre_model_wave = mod[0].data[mod[0].header['PREAXIS']]
        Fhp_model_wave = mod[0].data[mod[0].header['HEATAXIS']]

        model = Fpre_model_wave + Fhp_model_wave
        wl = mod[0].data[mod[0].header['WLAXIS']]

        #Add it to the edge object
        obj.add_spectra(job, wl, model)

    obj.saveObs(datapath = path)

    #Grab the exposure times for each time step and return it
    param = ascii.read(paramfile)

    tstep = np.array(param['time'])

    return tstep, obj
