import os, sys, numpy as np
import hoomd, hoomd.md as md
import gsd, gsd.hoomd, gsd.pygsd
import random
from hoomd import azplugins
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--temp',nargs='?',const='', type=int)
args = parser.parse_args()

################################################################################################
# ### Step 0: define system information, read parameters
################################################################################################

# SYSTEM SIZE
nx = 5 # Replicate system in x direction
ny = 5 # Replicate system in y direction
nz = 5 # Replicate system in z direction --> Total system size would be nx*ny*nz
# Please keep magnitudes of nx, ny and nz as close to each other
# as possible to avoid large replicated system box sizes in a specific direction

# SLAB DIMENSIONS
boxsize = 8.0 # The x and y dimensions of the slab configuration in nanometers
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

# GENERATE FOLDERS
folder = f'traj/poly50/hps/{args.temp}'
if not os.path.exists(folder):
    os.makedirs(folder)

################################################################################################
# ### Step 1: generate simulation system, resize simulation box
################################################################################################

bond_length = 0.38
chain_length = 50
box_length = bond_length * chain_length + 10

# Build HOOMD data structure for one single frame
s = gsd.hoomd.Snapshot()
s.particles.N = chain_length
s.particles.types = ['A']
s.particles.typeid = [0] * chain_length
s.particles.mass = [110] * chain_length
s.particles.charge = [0] * chain_length

# Build initial position as a linear chain
pos = []
for i in range(chain_length):
    # Change the z-coordinate to have a linear chain
    pos.append((0, 0, (i-int(chain_length/2))*bond_length))
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
f = gsd.hoomd.open(name=f'{folder}/start_{args.temp}.gsd', mode='wb')
f.append(s)
f.close()

hoomd.context.initialize("--mode=gpu")
system = hoomd.init.read_gsd(f'{folder}/start_{args.temp}.gsd')

# ### Replicate the single chain here. Remember total number of chains = nx*ny*nz
system.replicate(nx=nx, ny=ny, nz=nz)

# #### Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8368, r0=bond_length)

# #### Neighborlist and exclusions
nl = hoomd.md.nlist.cell()
nl.reset_exclusions(exclusions=['1-2', 'body'])

# #### Pairwise interactions
cal2j = 4.184
nb = azplugins.pair.ashbaugh(r_cut=0, nlist=nl)
nb.pair_coeff.set('A', 'A', lam=0.640, epsilon=0.2*cal2j, sigma=0.536, r_cut=2.0)

# #### Group Particles
all = hoomd.group.all()

# #### Set up integrator
hoomd.md.integrate.mode_standard(dt=resize_dt) # Time units in ps
kTinput = resize_T * 8.3144598 / 1000.
integrator = hoomd.md.integrate.langevin(group=all, kT=kTinput, seed=random.randint(1, 100000))
gamma = 0.01
integrator.set_gamma('A', gamma=110*gamma)

# #### Resize the box after replication to 15x15x15nm  
hoomd.update.box_resize(Lx=hoomd.variant.linear_interp([(0, system.box.Lx), (resize_steps-500, boxsize)]),
                        Ly=hoomd.variant.linear_interp([(0, system.box.Lx), (resize_steps-500, boxsize)]),
                        Lz=hoomd.variant.linear_interp([(0, system.box.Lx), (resize_steps-500, 15)]),
                        scale_particles=True)

# #### Output log file with box dimensions and restart file after box resizing
hoomd.analyze.log(filename=f'{folder}/resize_{args.temp}.log', quantities=['potential_energy', 'kinetic_energy', 'temperature', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'lx', 'ly', 'lz'], period=500, overwrite=True, header_prefix='#')
hoomd.dump.gsd(f'{folder}/resize_{args.temp}.gsd', period=500, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/resize_{args.temp}.dcd', period=500, group=all, overwrite=True)

# #### Run resizing simulation
hoomd.run(tsteps=resize_steps)

################################################################################################
# ### Step 2: extend the cubic box to slab shape and run simulation
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

f = gsd.pygsd.GSDFile(open(f'{folder}/resize_{args.temp}.gsd', 'rb'))
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
outfile = gsd.hoomd.open(f'{folder}/box2slab_{args.temp}.gsd', 'wb')
outfile.append(s)
outfile.close()

hoomd.context.initialize("--mode=gpu")
system = hoomd.init.read_gsd(f'{folder}/box2slab_{args.temp}.gsd')

n_steps = 1e7 # 100 nanoseconds

nl = hoomd.md.nlist.cell()

## Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8360, r0=bond_length)

## Nonbonded
nl.reset_exclusions(exclusions=['1-2', 'body'])
nb = azplugins.pair.ashbaugh(r_cut=0, nlist=nl)
nb.pair_coeff.set('A', 'A', lam=0.640, epsilon=0.2*cal2j, sigma=0.536, r_cut=2.0)

## Group Particles
all = hoomd.group.all()

## Set up integrator
hoomd.md.integrate.mode_standard(dt=production_dt) # Time units in ps
temp = 200 * 0.00831446
integrator = hoomd.md.integrate.langevin(group=all, kT=temp, seed=random.randint(1, 100000)) # Temp is kT/0.00831446
gamma = 0.01
integrator.set_gamma('A', gamma=110*gamma)
## Outputs
hoomd.analyze.log(filename=f'{folder}/extend_{args.temp}.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'pressure_xy', 'pressure_xz', 'pressure_yz', 'temperature', 'lx', 'ly', 'lz'], period=10000, overwrite=True, header_prefix='#')
hoomd.dump.gsd(f'{folder}/extend_{args.temp}.gsd', period=10000, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/extend_{args.temp}.dcd', period=10000, group=all, overwrite=True)

## Run simulation
hoomd.run(n_steps)

################################################################################################
# ### Step 3: run a production slab simulation using extend.gsd from previous step
################################################################################################

hoomd.context.initialize("--mode=gpu")
system = hoomd.init.read_gsd(f'{folder}/extend_{args.temp}.gsd')

n_steps = production_steps # 1 microseconds

nl = hoomd.md.nlist.cell()

## Bonds
harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('AA_bond', k=8360, r0=bond_length)

## Nonbonded
nl.reset_exclusions(exclusions=['1-2', 'body'])
nb = azplugins.pair.ashbaugh(r_cut=0, nlist=nl)
nb.pair_coeff.set('A', 'A', lam=0.640, epsilon=0.2*cal2j, sigma=0.536, r_cut=2.0)

## Group Particles
all = hoomd.group.all()

## Set up integrator
hoomd.md.integrate.mode_standard(dt=production_dt) # Time units in ps
temp = production_T * 0.00831446
integrator = hoomd.md.integrate.langevin(group=all, kT=temp, seed=random.randint(1, 100000)) # Temp is kT/0.00831446
gamma = 0.01
integrator.set_gamma('A', gamma=110*gamma)
## Outputs
hoomd.analyze.log(filename=f'{folder}/{production_T}.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'pressure_xy', 'pressure_xz', 'pressure_yz', 'temperature', 'lx', 'ly', 'lz'], period=100000, overwrite=False, header_prefix='#')
hoomd.dump.gsd(f'{folder}/{production_T}.gsd', period=1000000, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/{production_T}.dcd', period=100000, group=all, overwrite=False)

## Run simulation
hoomd.run_upto(production_steps)
################################################################################################
