import connector as con
import numpy as np
import matplotlib.pyplot as plt
import pdb
import EDGE as edge
from astropy.io import ascii, fits
import pickle
from astropy.table import Table
from glob import glob

'''
create_obj_template.py

PURPOSE:
    Template for phase 4: Creating trial objects from finished simulations/models and gets Q and M. This code will be generated by p3

AUTHHOR:
    Connor Robinson, June 19th, 2019
'''

model = 'test'
wttsname = 'twa7'

#Set up paths
pathfile = '/path/to/pathfile.dat'
tablefile = '/Users/Connor/Dropbox/Research/burst/sims/test/tablefile.dat'
shockparamfile = '/Users/Connor/Dropbox/Research/burst/sims/test/test/models/test/test_shock_params.dat'
modelpath = '/Users/Connor/Dropbox/Research/burst/sims/test/models/test/'
figpath = '/path/to/figures/'
p4path = '/path/to/p4/directory/'

phi0 = 0.0
alpha = 5
incs = [0,10,20,30,40,50,60,70,80,90]
pt = 0.2
ell_q = 0.3
ell_m = 4/24 
nzeros = 4
fracerr = 0.005

#Load in the wtts -- IF NOT USING EDGE, WILL NEED TO MODIFY THIS
path = ascii.read(pathfile)
wttspath = path['wttspath'][0]
wtts = edge.loadObs(wttsname, datapath = wttspath)
wtts_wl = wtts.spectra['HST']['wl']
wtts_flux = wtts.spectra['HST']['lFl']

#No longer need to modify below here
####################################
#Read in path file
wttsparamfile = path['wttspath'][0]+'stellar_params.dat'
wtable = ascii.read(wttsparamfile)

objname = model+'.pkl'
QMname = model+'_QM.dat'

#Grab the job numbers
jobnames = glob(modelpath+model+'_'+'?'*nzeros+'.fits')
jobs = np.sort(np.array([int(j[-(5+nzeros):-5]) for j in jobnames]))

wtable = ascii.read(wttsparamfile)
rwtts = wtable[wtable['objs'] == wttsname]['radius'][0]
dwtts = wtable[wtable['objs'] == wttsname]['dist'][0]

#Create the object
print('Creating Object')
obj = con.trial(jobs, model, modelpath, tablefile, shockparamfile, nzeros)
print('Adding photosphere')
obj.addPhot(wttsname, wtts_wl, wtts_flux, rwtts, dwtts)

iQM = []
lcs = {}

for inc in incs:
    name = 'i'+str(inc)
    print('Adding spot')
    obj.addSpot(inc, alpha, phi0, name, gamma = None)
    print('Making Spectra')
    obj.makeSpec(name)
    print('Making photometry')
    obj.makePhot(name, 'kepler', basepath = path['filters'][0])
    
    model = lcs[modelinc]['model']
    t = lcs[modelinc]['t']
    lc = lcs[modelinc]['lc']
    err = lcs[modelinc]['err']
    
    Q = con.getQ(t, lc, err, pt = pt, ell = ell_q, tau = 1, showplot = False, plotname = figpath+name+'_Q.pdf', gauss = True)
    M = con.getM(t, lc, ell_m, err, tau = 1, showplot = False, plotname = figpath+name+'_M.pdf', percentile = 5)
    
    #Add Q and M to the object
    obj.spot[incname]['Q'] = Q
    obj.spot[incname]['M'] = M
    
    iQM.append([inc, Q, M])

iQM = np.array(iQM)

#Write out Q and M
QMtab = Table(iQM, names = ['inc', 'Q', 'M'])
ascii.write(iQM, p4path+QMname)

#Save the object as a pickle file
pickle.dump(obj, open(p4path+objname, 'wb'))


