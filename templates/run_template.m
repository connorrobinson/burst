addpath /projectnb/bu-disks/connorr/fluid/fluid

nlabel = 1;
jobnum = [1];

%% Set up code to run in parallel
cluster = [0];%Set this to 1 if the code is running on the cluster (turns off progress bar)

%% General simulation parameters
% Parameters that are changed quite often
rdisk = [10]; % Outer grid radius
GAMMA = [10]; % Boct/Bdip
n = [4.5]; %Polytropic index n

% Mass of the star. NOTE! CHANGES THE DEPTH OF THE POTENTIAL WELL b
Mstar = [0.5 * 2e30];

% Parameters for constructing the coordinate system/time steps
Ncells = [1024]; % Number of grid cells
Nrun = [1e6]; % Number of time-steps
Ndump = [250]; % Store data every Ndump time-steps
index = [8]; % grid spacing setup

% Parameters that are only important in specific cases (e.g. odd BCs)
rho_star = [1e2]; % Surface density of the star
mdot = [.1]; % Accretion rate
isothermal = [0]; %If this is set to 1, then model is isothermal.
u_i = [-1e-5]; %Initial velocity for entire simulation (Probably -1 through 0)

% Parameters that almost never change
cs = [1]; % Isothermal sound speed
rho_disk = [1]; % Density of the disk
Z = [0.05]; % Height of the wall in stellar radii

%% Inner boundary condition
% 1 = simple outflow
% 2 = hard wall hydrostatic
% 3 = hydrostatic with decay
in_type = [1];

%% Outer boundary condition
% 0 = doesn't constrict outflow, simple inflow
% 1 = simple inflow, constant entropy (adiabatic variability)
% 2 = set accretion via mdot
% 3 = hydrostatic (doesn't work well)
% 4 = simple inflow, constant pressure (doesn't work well)
% 5 = Use an external file to explicitely set boundary conditions

out_type = [1];

%% Set up disk variability
% 0 = No variability
% 1 = Sinusoidal with: period = timescale, amplitude
% 2 = Step function: step occurs at delay, step height = rho0 * amplitude
% 3 = Top hat function: first step occurs at delay, width = timescale, height = amplitude

disk_variability = [1];

% Set up how the boundary condition will vary if variation is required
timescale = [8]; %THIS CODE IS SET UP TO BE IN PHYSICAL UNITS (PERIOD IS IN DAYS)
amplitude = [5];
delay = [15];

%% Constants + values for converting code units into physical units
% Radius of the star
R_sol = [1.5];%m
R = R_sol * 6.95508e8;
% Temperature at the disk due to UV heating
Tdisk = [10000];%K
% Proton mass
 mp = 1.67e-27;%kg    
% Boltzmann constant
kb = 1.38e-23;%J/K
% Gravitational constant
G = 6.67e-11;

%% Set up paths ect.
path = ['/projectnb/bu-disks/connorr/fluid/sine/'];
tagbase = ['testsine'];
saver = [1];

%% Set boundary condition file if necessary
boundary_file = 'None';

%% Run Code

%Calculate a few things for conversions + saving

if isothermal(1) ==1
    gamma = 1;
end

if isothermal(1) ~= 1
    gamma = (n+1)/n;
end

%Assuming solar metallicity
X = 0.7381;
Y = 0.2485;
Z = 0.0134;

mu = (2.*X + 3./4.*Y + 1./2.*Z).^-1;

Cs = sqrt(gamma .* kb .* Tdisk ./(mp*mu));
M = G.*Mstar./(R.*Cs.^2);

%Converts given period into sound crossing times
timescale_code = Cs./R.*timescale .* (60.*60.*24); 

%Runs oneDsolve with the parameters for this run
	[e_dump,t_dump,rb,ra,tha,rho_dump,u_dump,sigma_dot,g,difa,difb,h1a,...
	    h2a,h3a,h1b,dt,m_dump,s_dump,pa,pb,dr_dp,c_f] ...
	    = oneDsolve(M,cs,rho_disk,rdisk,Ncells,Nrun,Ndump,index,...
	    Z,mdot,rho_star,GAMMA,isothermal,n,u_i,timescale_code, ...
	    amplitude,in_type,out_type,disk_variability,delay,cluster, boundary_file);

tag = strcat(tagbase, sprintf('%03d', jobnum));

%Specify what to do with the data from the completed simulation

[saved] = writescript(saver, tag, path, nlabel, e_dump,t_dump,rb,ra,...
tha,rho_dump,u_dump,sigma_dot, g, ...
difa, difb, h1a, h2a, h3a, h1b,...
dt, m_dump, s_dump,pa,pb,dr_dp,c_f,...
M,cs,rho_disk,rdisk,Ncells,Nrun,...
Ndump,index,Z,mdot,rho_star,GAMMA,...
isothermal,n,u_i,timescale,amplitude,in_type,out_type,...
disk_variability,delay);

%make_vid
%get_dudt(u_dump,ra, t_dump, path, tag, saver)


