from astropy.io import fits, ascii
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import MovieWriter
from matplotlib.animation import FFMpegWriter
from collections import OrderedDict 
import pdb
import fluid as flu

'''

make_vid.py


PURPOSE:
    Makes a video of velocity and density information

AUTHOR: 
    Connor Robinson, August 26th, 2019

'''

# Load in the simulation

# simfile = '/Users/Connor/Dropbox/Research/burst/sims/var3/results/0001001--iso.mat'
# simfile = '/Users/Connor/Dropbox/Research/burst/sims/steady/results/0001001--iso.mat'
simfile = '/Users/Connor/Dropbox/Research/burst/sims/steady/results/0011001--iso.mat'

print('Loading simulation...')
sim = scipy.io.loadmat(simfile)
print('Simulation Loaded')




#########################

t_all = sim['t_dump'].flatten()
select = (t_all > 30) * (t_all < 35)

t = t_all[select]
u = sim['u_dump'][:,select]
rho = sim['rho_dump'][:,select]
ra = sim['ra'].flatten()[1:-1]
rb = sim['rb'].flatten()[1:-2]

GAMMA = sim['GAMMA'][0,0]
rdisk = sim['rdisk'][0,0]

tha = flu.get_theta(ra, rdisk, GAMMA, Z=0.05)
thb = flu.get_theta(rb, rdisk, GAMMA, Z=0.05)

hpa, hqa, h3a = flu.get_scale(ra, tha, GAMMA)
hpb, hqb, h3b = flu.get_scale(rb, thb, GAMMA)

Ab = hqb * h3b

#Get the mass accretion rate
mdot = []
for i, ti in enumerate(t):
    u_i = (u[1:,i] + u[:-1,i])/2
    mdot.append(Ab * u_i * rho[:,i])
mdot = np.array(mdot).T

#Get the acceleration using the gravity, centrifugal acceleration, the pressure support, and the advective term
b = sim['M'][0,0]
grav = flu.get_gravity(ra[1:-1],tha[1:-1],GAMMA,b)
rot = flu.get_rot(ra[1:-1], GAMMA, tha[1:-1], b, rdisk)


pb, q = flu.get_pq(rb,thb, GAMMA)
pa, q = flu.get_pq(ra,tha, GAMMA)


#a = []
bern = []
gP = []
ram = []
for i, ti in enumerate(t):
    gradP = 1/hpa[1:-1] * 2/(rho[:-1,i] + rho[1:,i]) * (rho[1:, i] - rho[:-1, i])/(pb[:-1] - pb[1:])
    ududp = -1/hpa[1:-1] * u[1:-1,i] * (u[:-2,i] - u[2:,i])/(pa[:-2] - pa[2:])
    
    gP.append(gradP)
    ram.append(ududp)
    
    # ududp = -1/hpa[1:-1] * u[1:-1,i] * (u[:-2,i] - u[1:-1,i])/(pa[:-2] - pa[1:-1])
    # ududp = -1/hpa[1:-1] * u[1:-1,i] * (u[2:,i] - u[1:-1,i])/(pa[2:] - pa[1:-1])
    
    #bern = u[1:-1,i]**2/2 - b/ra[1:-1] + b/(2*rdisk**4) * ra[1:-1]**3 - np.log(rho[1:,i])
    
    bern.append(u[1:-1,i]**2/2  - b/ra[1:-1] - b*ra[1:-1]**2*np.sin(tha[1:-1])**2/(rdisk**3*2) + np.log(rho[1:,i]))

# a = np.array(a).T
bern = np.array(bern).T / np.mean(bern)
gP = np.array(gP).T
ram = np.array(ram).T

#Set up figure

fig, axes = plt.subplots(2,2, figsize = [7, 7])
axes = axes.flatten()
xdata, ydata = [], []

#Create the lines + text
fig.subplots_adjust(hspace = 0, left = 0.12, wspace = 0.35, right = 0.98, top = 0.98)


ttext = axes[0].text(1.5, 0, 'time', color = 'k')

#Velocity panel
uln, = axes[0].plot([],[], color = 'k')
axes[0].set_xlim(1, 5)
axes[0].set_ylim(-38, 5)
axes[0].set_ylabel(r'$u \,[c_s]$')
axes[0].tick_params(axis = 'x', which = 'both', direction = 'in', labelbottom = False)

#Density panel
rln, = axes[2].plot([],[], color = 'k')
axes[2].set_yscale('log')
axes[2].set_xlim([1,5])
axes[2].set_ylim([1e-3, 15])
axes[2].set_ylabel(r'$\rho \,[\rho_{disk}]$')
axes[2].set_xlabel(r'$r \, [r{\star}]$')

#Mdot panel
mln, = axes[1].plot([],[], color = 'k')
axes[1].set_xlim(1, 5)
axes[1].set_ylim(-10,10)
axes[1].set_ylabel(r'$\dot{M}$')
axes[1].tick_params(axis = 'x', which = 'both', direction = 'in', labelbottom = False)

#Acceleration panel

#bln, = axes[3].plot([],[], color = 'k')

ramln, = axes[3].plot([],[], color = 'r')
pln, = axes[3].plot([],[], color = 'b')
tln, = axes[3].plot([],[],color = 'k')

axes[3].set_xlim([1,5])
axes[3].set_ylim([-100, 100])
#axes[3].set_ylabel(r'$\phi_{bern}$')
axes[3].set_ylabel(r'$a$')

axes[3].plot(ra[1:-1], grav, color = 'magenta')
axes[3].plot(ra[1:-1], rot, color = 'g')

axes[3].set_xlabel(r'$r \, [r{\star}]$')

def update(frame, ra, rb, u, rho, mdot, bern, gP, ram, t, n, rot, grav):
    
    print(str(frame+1)+'/'+str(len(n)))
    
    mln.set_data(rb, mdot[:,frame])
    uln.set_data(ra, u[:,frame])
    rln.set_data(rb, rho[:,frame])
    
    
    pln.set_data(ra[1:-1], gP[:, frame])
    ramln.set_data(ra[1:-1], ram[:, frame])
    tln.set_data(ra[1:-1], gP[:, frame] +  ram[:, frame] + rot + grav)
    
    # bln.set_data(ra[1:-1], bern[:,frame])
    
    
    ttext.set_text('{:.2f}'.format(t[frame]))
    
    return uln, rln

n = np.arange(len(t))

ani = FuncAnimation(fig, update, frames = n, blit=True, fargs = [ra, rb, u, rho, mdot, bern, gP, ram, t, n, rot, grav])

ani.save('basic_animation.mp4', fps=60)

