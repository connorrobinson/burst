import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import imp

'''
runshock_template

PURPOSE:
    This is a template that will be created to run Phase III under the default values
    This is not just a function, because this must be run AFTER tbe simulations are run on the cluster. 
    
    This assumes that the shock models are run locally. 
    
AUTHOR:
    Connor Robinson, January 30th, 2019
'''

#Define variables. Parsed and replaced automatically, so do not change formatting. 
tablefile = '/Users/Connor/Dropbox/Research/burst/example/table.dat'
model = 'model'
simulation = '/Users/Connor/Dropbox/Research/burst/example/example001--iso.mat'
outpath = '/Users/Connor/Dropbox/Research/burst/example/'
pathfile = '/Users/connor/Dropbox/Research/burst/code/paths/scc_paths.dat'

#Variables for creating the phase 4 file.
jobid = 'jobid'
p4path = '/path/to/big/grid/of/p4/models/'

#Open up the table with values
table = ascii.read(tablefile)

#Set the number of threads
nthreads = 3

#Define model charachteristics
TSTAR = table['teff'][0]
DISTANCE = table['dist'][0]
RADIUS = table['R_sol'][0]
Tdisk = table['Tdisk'][0]
n0 = table['n0'][0]
tstart = table['tstart'][0]
tend = table['tend'][0]
cadence = table['cadence'][0]

path = ascii.read(pathfile)

#Load in the connector module
con = imp.load_source('connector', path['PYTHONCODE'][0]+'connector.py')

p4_template_job = path['p4_template_job'][0]

#Construct model for the simulations
con.connector(model, simulation, Tdisk, DISTANCE, TSTAR, RADIUS, n0, tstart, tend, cadence,\
              runmodel = False,\
              outpath = outpath,\
              composition = 'solar',\
              templatespectra = path['templatespectra'][0],\
              shocktemplate = path['shocktemplate'][0],\
              runalltemplate = path['runalltemplate'][0],\
              DIRPROG = path['DIRPROG'][0],\
              DIRDAT = path['DIRDAT'][0],\
              BASEPATH = path['BASEPATH'][0],\
              CTFILE = path['CTFILE'][0],\
              COOLTAG = 'cooling',\
              CLOUDY = path['CLOUDY'][0],\
              OPCFILE = path['OPCFILE'][0],\
              LOCALPYTHONPATH = path['LOCALPYTHONPATH'][0],\
              nzeros = 4)

#Create the phase 4 job.
con.create_p4job(jobid, p4path, model, p4_template_job)
