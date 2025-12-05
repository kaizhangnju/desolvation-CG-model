import os, sys, numpy as np
import hoomd, hoomd.md as md
import gsd, gsd.hoomd, gsd.pygsd
import random
from decimal import Decimal
from argparse import ArgumentParser

def hps_desolvation(r, rmax, rmin, r_b, r_w, var_b, var_w, lam, sigma, e_h, e_b, e_w):
	lj_potential = 4 * e_h * ((sigma / r)**12 - (sigma / r)**6)
	lj_potential_diff = 24 * e_h / r * (2 * (sigma / r)**12 - (sigma / r)**6)
	if r <= 2**(1.0 / 6.0) * sigma:
		V = lj_potential + (1 - lam) * e_h\
          + e_b * np.exp(-(r-r_b)**2 / var_b)\
          - e_w * np.exp(-(r-r_w)**2 / var_w)
		F = lj_potential_diff\
          + e_b * 2 * (r-r_b) / var_b * np.exp(-(r-r_b)**2 / var_b)\
          - e_w * 2 * (r-r_w) / var_w * np.exp(-(r-r_w)**2 / var_w)
	else:
		V = lam * lj_potential\
          + e_b * np.exp(-(r-r_b)**2 / var_b)\
          - e_w * np.exp(-(r-r_w)**2 / var_w)
		F = lam * lj_potential_diff\
          + e_b * 2 * (r-r_b) / var_b * np.exp(-(r-r_b)**2 / var_b)\
          - e_w * 2 * (r-r_w) / var_w * np.exp(-(r-r_w)**2 / var_w)
	return (V, F)

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--alpha_b',nargs='?',const='', type=str)
parser.add_argument('--alpha_w',nargs='?',const='', type=str)
parser.add_argument('--eps_scale',nargs='?',const='', type=str)
parser.add_argument('--temp',nargs='?',const='', type=int)
args = parser.parse_args()

seq = {'R':'ARG','H':'HIS','K':'LYS','D':'ASP','E':'GLU',
       'S':'SER','T':'THR','N':'ASN','Q':'GLN','C':'CYS',
       'U':'SEC','G':'GLY','P':'PRO','A':'ALA','V':'VAL',
       'I':'ILE','L':'LEU','M':'MET','F':'PHE','Y':'TYR',
       'W':'TRP'}

# SYSTEM SIZE
nx = 5 # Replicate system in x direction 
ny = 5 # Replicate system in y direction 
nz = 4 # Replicate system in z direction --> Total system size would be nx*ny*nz
# Please keep magnitudes of nx, ny and nz as close to each other 
# as possible to avoid large replicated system box sizes in a specific direction 

# SLAB DIMENSIONS
boxsize = 15.0 # The x and y dimensions of the slab configuration in nanometers
slab_z_length = 280.0 # The z dimension of the slab configuration in nanometers

# RESIZING RUN PARAMETERS
resize_T = 300 # Temperature for resizing run in Kelvin
resize_steps = 50000 # Number of steps (>500) to be used for resizing the box to boxsize set previously
if resize_steps < 500:
    resize_steps = 501
resize_dt = 0.01 # Time step in picoseconds for box resizing 

# PRODUCTION RUN PARAMETERS
production_dt = 0.01 # Time step for production run in picoseconds
production_steps = 100000000 # Total number of steps 
production_T = args.temp # Temperature for production run in Kelvin

# MODIFIED PAIRWISE FORCE MODEL
protein = args.name
cal2j = Decimal(4.184) # 1 cal = 4.184 J
e_ratio = Decimal(args.eps_scale)
eb_ratio = Decimal(args.alpha_b)
ew_ratio = Decimal(args.alpha_w)
eh = (Decimal('0.2') * e_ratio).normalize()
eb = (Decimal('0.2') * e_ratio * eb_ratio).normalize()
ew = (Decimal('0.2') * e_ratio * ew_ratio).normalize()

# GENERATE FOLDERS
fileroot = str(production_T)
folder = f'traj/{protein}/des_{eh}_{eb}_{ew}/{production_T}'
if not os.path.exists(folder):
    os.makedirs(folder)

# ## 1.0 Read sequence data and ff parameters

# #### 1.1 Read sequence and force field parameters
# ##### Input parameters for all the amino acids (force field)
ff_para = '../data/stats_module.dat'
aalist = {}
with open(ff_para, 'r') as fid:
    for i in fid:
        if i[0] != '#':
            tmp = i.rsplit()
            aalist[tmp[0]] = np.loadtxt(tmp[1:], dtype=float)
aakeys = list(aalist.keys())

# ##### Read one letter amino acid sequence from file and translate the
# entire sequence into a number code according to the order in 'aakeys'
filein = f'../data/{protein}.dat'
chain_id = []
chain_mass = []
chain_charge = []
with open(filein, 'r') as fid:
    for i in fid:
        if i[0] != '#':
            for j in i:
                if j in seq:
                    iname = seq[j]
                    chain_id.append(aakeys.index(iname))
                    chain_mass.append(aalist[iname][0])
                    chain_charge.append(aalist[iname][1])

# This translates each amino acid type into a number, which will be used in HOOMD
aamass = []
aacharge = []
aaradius = []
aahps = []
for i in aakeys:
    aamass.append(aalist[i][0])
    aacharge.append(aalist[i][1])
    aaradius.append(aalist[i][2])
    aahps.append(aalist[i][3])

bond_length = 0.38
chain_length = len(chain_id)
box_length = bond_length * chain_length + 10

# #### 1.2 Now we can build HOOMD data structure for one single frame
s = gsd.hoomd.Snapshot()
s.particles.N = chain_length
s.particles.types = aakeys
s.particles.typeid = chain_id
s.particles.mass = chain_mass
s.particles.charge = chain_charge

# Build initial position as a linear chain
pos = []
for i in range(len(chain_id)):
    # Change the z-coordinate to have a linear chain
    pos.append((0, 0, (i-int(len(chain_id)/2))*bond_length))
pos = np.array(pos)
s.particles.position = pos

# Initialize bond
nbonds = chain_length - 1
s.bonds.N = nbonds
s.bonds.types = ['AA_bond']
s.bonds.typeid = [0] * (nbonds)
bond_pairs = np.zeros((nbonds, 2), dtype=int)
for i in range(nbonds):
    bond_pairs[i, :] = np.array([i, i+1])
s.bonds.group = bond_pairs

# Box size
s.configuration.dimensions = 3
s.configuration.box = [box_length, box_length, box_length, 0, 0, 0]
s.configuration.step = 0

# #### 1.3 Write intial singe chain gsd file
startfile = f'{folder}/start_{fileroot}.gsd'
f = gsd.hoomd.open(name=startfile, mode='wb')
f.append(s)
f.close()
################################################################################################
# ## start.gsd contains one single chain of the given protein 
# ----------------------------------------------------------------------------------------------
# ## 2.0 Replicate single chain system to given number of chains for a slab 
################################################################################################
hoomd.context.initialize("--mode=gpu")
system = hoomd.init.read_gsd(startfile)

# ### Replicate the single chain here. Remember total number of chains = nx*ny*nz
system.replicate(nx=nx,ny=ny,nz=nz) 

# #### Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8368, r0=bond_length)

# #### Neighborlist and exclusions
nl = hoomd.md.nlist.cell()
nl.reset_exclusions(exclusions=['1-2', 'body'])

# #### Pairwise interactions
var_b, var_w = 0.005, 0.005
nb = hoomd.md.pair.table(width=1000, nlist=nl)
for i in aakeys:
    for j in aakeys:
        lam = (aalist[i][3]+aalist[j][3])/2.
        sigma = (aalist[i][2]+aalist[j][2])/10./2.
        r_m = 2**(1. / 6.) * sigma
        r_b = r_m + 0.15
        r_w = r_m + 0.30
        e_h = float(eh * cal2j)
        e_b = float(eb * cal2j)
        e_w = float(ew * cal2j)
        nb.pair_coeff.set(i, j, func=hps_desolvation, rmin=0.1, rmax=sigma*3,
                          coeff=dict(r_b=r_b, r_w=r_w, var_b=var_b, var_w=var_w,
                          lam=lam, sigma=sigma, e_h=e_h, e_b=e_b, e_w=e_w))
        
# #### Electrostatics
yukawa = hoomd.md.pair.yukawa(r_cut=0.0, nlist=nl)
for i, atom1 in enumerate(aakeys):
    for j, atom2 in enumerate(aakeys):
        yukawa.pair_coeff.set(atom1, atom2, epsilon=aalist[atom1][1]*aalist[atom2][1]*1.73136, kappa=1.0, r_cut=3.5) 

# #### Group Particles
all = hoomd.group.all()

# #### Set up integrator
hoomd.md.integrate.mode_standard(dt=resize_dt) # Time units in ps
kTinput = resize_T * 8.3144598 / 1000.
integrator = hoomd.md.integrate.langevin(group=all, kT=kTinput, seed=random.randint(1, 100000))

# #### Resize the box after replication to 15x15x15nm
press_z = 10.0
Lx_press = hoomd.variant.linear_interp([(0, system.box.Lx), (resize_steps-500, boxsize)])
Ly_press = hoomd.variant.linear_interp([(0, system.box.Ly), (resize_steps-500, boxsize)])
Lz_press = hoomd.variant.linear_interp([(0, system.box.Lz), (resize_steps-500, press_z)])
hoomd.update.box_resize(Lx=Lx_press, Ly=Ly_press, Lz=Lz_press, scale_particles=True)
for cnt, i in enumerate(aakeys):
    integrator.set_gamma(i, gamma=aamass[cnt]/1000.0)

# #### Output log file with box dimensions and restart file after box resizing
hoomd.analyze.log(filename=f'{folder}/resize_{fileroot}.log', quantities=['potential_energy', 'kinetic_energy', 'temperature', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'lx', 'ly', 'lz'], period=500, overwrite=True, header_prefix='#')
hoomd.dump.gsd(f'{folder}/resize_{fileroot}.gsd', period=500, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/resize_{fileroot}.dcd', period=500, group=all, overwrite=True)

# #### Run resizing simulation
hoomd.run(tsteps=resize_steps)
################################################################################################
# ## resize.gsd contains replicated system with 100 chains and box size of 15x15x15 nm 
# ----------------------------------------------------------------------------------------------
# ## 3.0 Extend 15nm cube to 15x15x280nm slab 
################################################################################################
def extend(s):
    boxdim = s.configuration.box[:3]
    zmin, zmax, dz = -boxdim[2]/2., boxdim[2]/2., boxdim[2]
    pos1 = s.particles.position
    pos = pos1.copy()
    skip = 0
    ncomp = 1
    for k in range(ncomp):
        nchain = int(s.particles.N/chain_length)
        nres = chain_length
        for i in range(nchain):
            mol_coord = pos[i*nres+skip:(i+1)*nres+skip, 2]
            for j in range(1, nres):
                dist2 = (mol_coord[j] - mol_coord[j-1])**2
                if dist2 > 8:
                    excess = np.sign(mol_coord[j] - mol_coord[j-1]) * dz
                    mol_coord[j] = mol_coord[j] - excess 
                com = np.mean(mol_coord)
                if com < zmin:
                    mol_coord += dz
                elif com > zmax:
                    mol_coord -= dz
            pos[i*nres+skip:(i+1)*nres+skip, 2] = mol_coord
        skip += nchain * nres
    return pos

f = gsd.pygsd.GSDFile(open(f'{folder}/resize_{fileroot}.gsd', 'rb'))
t = gsd.hoomd.HOOMDTrajectory(f)
s1 = t[0]
s = gsd.hoomd.Snapshot()
s.particles.N = s1.particles.N
s.particles.types = s1.particles.types 
s.particles.typeid = s1.particles.typeid 
s.particles.mass = s1.particles.mass
s.particles.charge = s1.particles.charge
s.particles.position = extend(s1)
s.bonds.N = s1.bonds.N
s.bonds.types = s1.bonds.types
s.bonds.typeid = s1.bonds.typeid
s.bonds.group = s1.bonds.group
s.configuration.box = s1.configuration.box
s.configuration.dimensions = 3
s.configuration.box = [s1.configuration.box[0], s1.configuration.box[1], slab_z_length, 0, 0, 0] 
s.configuration.step = 0
outfile = gsd.hoomd.open(f'{folder}/box2slab_{fileroot}.gsd', 'wb')
outfile.append(s)
outfile.close()
################################################################################################
# ### Minimized slab formed and saved in minimize.gsd
#-----------------------------------------------------------------------------------------------
# ## 4.0. Run a production slab simulation using minimize.gsd from previous step
################################################################################################
hoomd.context.initialize()
system = hoomd.init.read_gsd(f'{folder}/box2slab_{fileroot}.gsd')

n_steps = production_steps # 1 microseconds

nl = hoomd.md.nlist.cell()

## Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8360, r0=0.381)

## Nonbonded
var_b, var_w = 0.005, 0.005
nb = hoomd.md.pair.table(width=1000, nlist=nl)
for i in aakeys:
    for j in aakeys:
        lam = (aalist[i][3]+aalist[j][3])/2.
        sigma = (aalist[i][2]+aalist[j][2])/10./2.
        r_m = 2**(1. / 6.) * sigma
        r_b = r_m + 0.15
        r_w = r_m + 0.30
        e_h = float(eh * cal2j)
        e_b = float(eb * cal2j)
        e_w = float(ew * cal2j)
        nb.pair_coeff.set(i, j, func=hps_desolvation, rmin=0.1, rmax=sigma*3,
                          coeff=dict(r_b=r_b, r_w=r_w, var_b=var_b, var_w=var_w,
                          lam=lam, sigma=sigma, e_h=e_h, e_b=e_b, e_w=e_w))

## Electrostatics
yukawa = hoomd.md.pair.yukawa(r_cut=0.0, nlist=nl)
for i, atom1 in enumerate(aakeys):
    for j, atom2 in enumerate(aakeys):
        yukawa.pair_coeff.set(atom1, atom2, epsilon=aalist[atom1][1]*aalist[atom2][1]*1.73136, kappa=1.0, r_cut=3.5)

## Group Particles
all = hoomd.group.all()

## Set up integrator
hoomd.md.integrate.mode_standard(dt=production_dt) # Time units in ps
temp = production_T * 0.00831446
integrator = hoomd.md.integrate.langevin(group=all, kT=temp, seed=random.randint(1, 100000)) # Temp is kT/0.00831446
for cnt, i in enumerate(aakeys):
    integrator.set_gamma(i, gamma=aamass[cnt]/1000.0)
## Outputs
hoomd.analyze.log(filename=f'{folder}/{fileroot}.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'pressure_xy', 'pressure_xz', 'pressure_yz', 'temperature', 'lx', 'ly', 'lz'], period=100000, overwrite=False, header_prefix='#')
hoomd.dump.gsd(f'{folder}/{fileroot}.gsd', period=1000000, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/{fileroot}.dcd', period=100000, group=all, overwrite=False)

## Run simulation
hoomd.run_upto(production_steps)
########################################################################################################
