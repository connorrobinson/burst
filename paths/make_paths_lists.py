from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

#Create local paths
local = np.array([\
         ['cttspath', '/Users/Connor/Dropbox/Research/shock/data/ctts/'],\
         ['templatespectra','/Users/Connor/Dropbox/Research/burst/code/templates/template_spectra.dat'],\
         ['shocktemplate','/Users/Connor/Dropbox/Research/burst/code/templates/burst_template'],\
         ['runalltemplate','/Users/Connor/Dropbox/Research/burst/code/templates/runall_template'],\
         ['p4_template_job', '/Users/Connor/Dropbox/Research/burst/code/templates/p4_template_job'],\
         ['wttspath','/Users/Connor/Dropbox/Research/burst/code/wtts/'],\
         ['filters', '/Users/Connor/Dropbox/Research/burst/resources/Filters/'],\
         ['DIRPROG','/Users/Connor/Dropbox/Research/shock/shockmodel/PROGRAMS/'],\
         ['DIRDAT','/Users/Connor/Dropbox/Research/shock/shockmodel/DATAFILES/'],\
         ['BASEPATH','/Users/Connor/Dropbox/Research/shock/code/cloudy_code/models/'],\
         ['CTFILE','/Users/Connor/Dropbox/Research/shock/code/cloudy_code/coolinggrid.txt'],\
         ['CLOUDY','/Users/Connor/Desktop/Research/cloudy/c17.00/source/cloudy.exe'],\
         ['OPCFILE','/Users/Connor/Dropbox/Research/shock/code/cloudy_code/opacitygrid.txt'],\
         ['LOCALPYTHONPATH', '/Users/Connor/anaconda/bin/python'],\
         ['PYTHONCODE', '/Users/Connor/Dropbox/Research/burst/code/']])
localtable = Table(local[:,1], names = local[:,0])

localtable.write('local_paths.dat', format = 'ascii', overwrite = True)


scc = np.array([\
         ['cttspath', '/Users/Connor/Dropbox/Research/shock/data/ctts/'],\
         ['templatespectra','/projectnb/bu-disks/connorr/burst/code/templates/template_spectra.dat'],\
         ['shocktemplate','/projectnb/bu-disks/connorr/burst/code/templates/burst_template'],\
         ['runalltemplate','/projectnb/bu-disks/connorr/burst/code/templates/runall_template'],\
         ['p4_template_job', '/projectnb/bu-disks/connorr/burst/code/templates/p4_template_job'],\
         ['wttspath','/projectnb/bu-disks/connorr/burst/code/wtts/'],\
         ['filters','/projectnb/bu-disks/connorr/burst/resources/Filters/'],\
         ['DIRPROG','/projectnb/bu-disks/connorr/SHOCK/shockmodel/PROGRAMS'],\
         ['DIRDAT','/projectnb/bu-disks/connorr/SHOCK/shockmodel/DATAFILES'],\
         ['BASEPATH','/project/bu-disks/shared/SHOCK/PREPOST/models/'],\
         ['CTFILE','/project/bu-disks/shared/SHOCK/PREPOST/coolinggrid.txt'],\
         ['CLOUDY','/projectnb/bu-disks/connorr/cloudy/c17.00/source/cloudy.exe'],\
         ['OPCFILE','/project/bu-disks/shared/SHOCK/PREPOST/opacitygrid.txt'],\
         ['LOCALPYTHONPATH', None],\
         ['PYTHONCODE', '/projectnb/bu-disks/connorr/burst/code/']])

scctable = Table(scc[:,1], names = scc[:,0])

scctable.write('scc_paths.dat', format = 'ascii', overwrite = True)





