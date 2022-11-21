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

model = '0001'
wttsname = 'recx1'
band = 'U'

#Set up paths
pathfile = '/Users/Connor/Dropbox/Research/burst/code/paths/local_paths.dat'
tablefile = '/Users/Connor/Dropbox/Research/burst/sims/var3/0001/support/0001_params.dat'
shockparamfile = '/Users/Connor/Dropbox/Research/burst/sims/var3/0001/models/0001/0001_shock_params.dat'
modelpath = '/Users/Connor/Dropbox/Research/burst/sims/var3/0001/models/0001/'
figpath = '/Users/Connor/Dropbox/Research/burst/sims/var3/code/enginework/'
p4path = '/Users/Connor/Dropbox/Research/burst/sims/var3/code/enginework/'

phi0 = 0.0
ftrunc = 0.5
width = 20
incs = [60]
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
Rwtts = wtable[wtable['objs'] == wttsname]['radius'][0]
Dwtts = wtable[wtable['objs'] == wttsname]['dist'][0]

#Load in the wtts flux for the requested band
wtts_fluxtab = ascii.read(wttspath+wttsname+'phot.dat')
wttsbandflux = wtts_fluxtab['flux'][wtts_fluxtab['band'] == band][0]

#Create the object
print('Creating Object')
obj = con.trial(jobs, model, modelpath, tablefile, shockparamfile, nzeros)

#Create the photometry from the spectrum
print('Convering Spectrum to Photometry')
obj.spec2phot(band, wttsbandflux, Rwtts, Dwtts, bandpath = path['filters'][0])

#Add the photometry (for testing purposes)
print('Adding photosphere')
obj.addPhot(wttsname, wtts_wl, wtts_flux, Rwtts, Dwtts)

iQM = []
lcs = {}

for inc in incs:
    name = 'i'+str(inc)
    print('Adding spot')
    obj.addStrip(inc, ftrunc, width, phi0, name)
    print('Making Photometry')
    obj.makeLC(band, name)
    raw = obj.spot[name][band]
    
    t = obj.time.flatten()
    lc = raw + np.random.normal(scale = fracerr * np.median(raw), size = len(raw))
    err = fracerr * np.median(lc)
    
    lcs[name] = {'lc':lc, 't':t, 'err':err, 'model':model}
    
    Q = con.getQ(t, lc, err, pt = pt, ell = ell_q, tau = 1, showplot = False, plotname = figpath+name+'_Q.pdf', gauss = True)
    M = con.getM(t, lc, ell_m, err, tau = 1, showplot = False, plotname = figpath+name+'_M.pdf', percentile = 5)
    
    #Add Q and M to the object
    obj.spot[name]['Q'] = Q
    obj.spot[name]['M'] = M
    
    iQM.append([inc, Q, M])

iQM = np.array(iQM)

#Write out Q and M
QMtab = Table(iQM, names = ['inc', 'Q', 'M'])
ascii.write(iQM, p4path+QMname)

#Save the object as a pickle file
pickle.dump(obj, open(p4path+objname, 'wb'))


