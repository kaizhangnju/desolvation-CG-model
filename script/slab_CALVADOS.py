import sys, os
import hoomd
import hoomd.md
import itertools
import pandas as pd
import numpy as np
import gsd, gsd.hoomd, gsd.pygsd
from decimal import Decimal
from hoomd import azplugins
from argparse import ArgumentParser

def genParams(r, prot, Tsim):
    RT = 8.3145 * Tsim * 1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(Tsim)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    
    fasta = prot.fasta
    
    # Set the charge on HIS based on the pH of the protein solution? Not needed if pH=7.4
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['X','q'] = r.loc[fasta[0],'q'] + 1.
    r.loc['X','MW'] = r.loc[fasta[0],'MW'] + 2.
    fasta[0] = 'X'
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['Z','q'] = r.loc[fasta[-1],'q'] - 1.
    r.loc['Z','MW'] = r.loc[fasta[-1],'MW'] + 16.
    fasta[-1] = 'Z'
    
    # Calculate the prefactor for the Yukawa potential
    qq = pd.DataFrame(r.q.values*r.q.values.reshape(-1,1),index=r.q.index,columns=r.q.index)
    yukawa_eps = qq*lB*RT
    types = list(np.unique(fasta))
    pairs = np.array(list(itertools.combinations_with_replacement(types,2)))
    return yukawa_kappa, yukawa_eps, types, pairs, fasta, r

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--temp',nargs='?',const='', type=int)
args = parser.parse_args()

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
production_T = int(args.temp) # Temperature for production run in Kelvin

# MODIFIED PAIRWISE FORCE MODEL
protein = args.name
cal2j = Decimal(4.184) # 1 cal = 4.184 J

# GENERATE TRAJECTORY FOLDERS
fileroot = str(production_T)
folder = f'traj/{protein}/CALVADOS/{production_T}'
if not os.path.exists(folder):
    os.makedirs(folder)

# ## 1.0 Read sequence data and ff parameters

# #### 1.1 Read sequence and force field parameters
# ##### Input parameters for all the amino acids (force field)
residues = pd.read_csv('../data/residues.csv').set_index('one',drop=False)
proteins = pd.read_pickle('../data/allproteins.pkl').astype(object)
prot = proteins.loc[protein]
yukawa_kappa, yukawa_eps, types, pairs, fasta, residues = genParams(residues,prot,production_T)
sigmamap = pd.DataFrame((residues.sigmas.values+residues.sigmas.values.reshape(-1,1))/2,
                        index=residues.sigmas.index,columns=residues.sigmas.index)
lambdamap = pd.DataFrame((residues.lambdas.values+residues.lambdas.values.reshape(-1,1))/2,
                        index=residues.lambdas.index,columns=residues.lambdas.index)

# #### 1.2 Now we can build HOOMD data structure for one single frame
hoomd.context.initialize("--mode=gpu")

N_res = prot.N
L = (N_res - 1) * .38 + 4

snapshot = hoomd.data.make_snapshot(N=N_res,
                                    box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L),
                                    particle_types=types,
                                    bond_types=['polymer'])

snapshot.bonds.resize(N_res-1)

snapshot.particles.position[:] = [[0,0,(i-N_res/2.)*.38] for i in range(N_res)]
snapshot.particles.typeid[:] = [types.index(a) for a in fasta]
snapshot.particles.mass[:] = [residues.loc[a].MW for a in fasta]

snapshot.bonds.group[:] = [[i,i+1] for i in range(N_res-1)]
snapshot.bonds.typeid[:] = [0] * (N_res-1)
################################################################################################
# ## start.gsd contains one single chain of the given protein 
# ----------------------------------------------------------------------------------------------
# ## 2.0 Replicate single chain system to given number of chains for a slab 
################################################################################################
system = hoomd.init.read_snapshot(snapshot)

# ### Replicate the single chain here. Remember total number of chains = nx*ny*nz
system.replicate(nx=nx,ny=ny,nz=nz) 

# #### Bonds
hb = hoomd.md.bond.harmonic()
hb.bond_coeff.set('polymer', k=8033.0, r0=0.38)

# #### Neighborlist and exclusions
nl = hoomd.md.nlist.cell()

# #### Pairwise interactions and Electrostatics
cutoff = 2.4
lj_eps = 4.184*.2
ah = azplugins.pair.ashbaugh(r_cut=cutoff, nlist=nl)
yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
for a, b in pairs:
    ah.pair_coeff.set(a, b, lam=lambdamap.loc[a,b], epsilon=lj_eps, sigma=sigmamap.loc[a,b], r_cut=cutoff)
    yukawa.pair_coeff.set(a, b, epsilon=yukawa_eps.loc[a,b], kappa=yukawa_kappa, r_cut=4.)

yukawa.set_params(mode='shift')
nl.reset_exclusions(exclusions = ['bond'])

# #### Group Particles
all = hoomd.group.all()

# #### Set up integrator
RT = 8.3145*production_T*1e-3
integrator_mode = hoomd.md.integrate.mode_standard(dt=resize_dt)
integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100))

# #### Resize the box after replication to 15x15x15nm
press_z = 10.0
Lx_press = hoomd.variant.linear_interp([(0, system.box.Lx), (resize_steps-500, boxsize)])
Ly_press = hoomd.variant.linear_interp([(0, system.box.Ly), (resize_steps-500, boxsize)])
Lz_press = hoomd.variant.linear_interp([(0, system.box.Lz), (resize_steps-500, press_z)])
hoomd.update.box_resize(Lx=Lx_press, Ly=Ly_press, Lz=Lz_press, scale_particles=True)
for a in types:
    integrator.set_gamma(a, residues.loc[a].MW/100)

# #### Output log file with box dimensions and restart file after box resizing
hoomd.analyze.log(filename=f'{folder}/resize_{production_T}.log', quantities=['potential_energy', 'kinetic_energy', 'temperature', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'lx', 'ly', 'lz'], period=500, overwrite=True, header_prefix='#')
hoomd.dump.gsd(f'{folder}/resize_{production_T}.gsd', period=500, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/resize_{production_T}.dcd', period=500, group=all, overwrite=True)

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
        nchain = int(s.particles.N/len(fasta))
        nres = len(fasta)
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

f = gsd.pygsd.GSDFile(open(f'{folder}/resize_{production_T}.gsd', 'rb'))
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
outfile = gsd.hoomd.open(f'{folder}/box2slab_{production_T}.gsd', 'wb')
outfile.append(s)
outfile.close()
################################################################################################
# ### Minimized slab formed and saved in minimize.gsd
#-----------------------------------------------------------------------------------------------
# ## 4.0. Run a production slab simulation using minimize.gsd from previous step
################################################################################################
hoomd.context.initialize("--mode=gpu")
system = hoomd.init.read_gsd(f'{folder}/box2slab_{production_T}.gsd')

n_steps = production_steps # 1 microseconds

nl = hoomd.md.nlist.cell()

## Bonds
hb = hoomd.md.bond.harmonic()
hb.bond_coeff.set('polymer', k=8033.0, r0=0.38)

## Nonbonded and Electrostatics
cutoff = 2.4
lj_eps = 4.184*.2
ah = azplugins.pair.ashbaugh(r_cut=cutoff, nlist=nl)
yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
for a, b in pairs:
    ah.pair_coeff.set(a, b, lam=lambdamap.loc[a,b], epsilon=lj_eps, sigma=sigmamap.loc[a,b], r_cut=cutoff)
    yukawa.pair_coeff.set(a, b, epsilon=yukawa_eps.loc[a,b], kappa=yukawa_kappa, r_cut=4.)

yukawa.set_params(mode='shift')
nl.reset_exclusions(exclusions = ['bond'])

## Group Particles
all = hoomd.group.all()

## Set up integrator
integrator_mode = hoomd.md.integrate.mode_standard(dt=production_dt)
integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100))
for a in types:
    integrator.set_gamma(a, residues.loc[a].MW/100)

## Outputs
hoomd.analyze.log(filename=f'{folder}/{fileroot}.log', quantities=['potential_energy', 'pressure_xx', 'pressure_yy', 'pressure_zz', 'pressure_xy', 'pressure_xz', 'pressure_yz', 'temperature', 'lx', 'ly', 'lz'], period=100000, overwrite=False, header_prefix='#')
hoomd.dump.gsd(f'{folder}/{fileroot}.gsd', period=1000000, group=all, truncate=True)
hoomd.dump.dcd(f'{folder}/{fileroot}.dcd', period=100000, group=all, overwrite=False)

## Run simulation
hoomd.run_upto(production_steps)
########################################################################################################
