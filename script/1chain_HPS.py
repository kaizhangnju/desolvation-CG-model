import os, time
import numpy as np
import hoomd, hoomd.md as md
import gsd, gsd.hoomd, gsd.pygsd
import pandas as pd
from hoomd import azplugins
from argparse import ArgumentParser

def run_single_chain_simulation(startfile, folder, aalist, aakeys, aamass, T_sim):

    N_res = len(fasta)
    N_save = 3000 if N_res < 100 else int(np.ceil(3e-4*N_res**2)*1000)
    N_steps = 1100 * N_save

    system = hoomd.init.read_gsd(startfile) 

    # Bonds
    harmonic = hoomd.md.bond.harmonic()
    harmonic.bond_coeff.set('AA_bond', k=8368, r0=bond_length)

    # Neighborlist and exclusions
    nl = hoomd.md.nlist.cell()
    nl.reset_exclusions(exclusions=['1-2', 'body'])

    # Pairwise interactions
    nb = azplugins.pair.ashbaugh(r_cut=0, nlist=nl)
    for i in aakeys:
        for j in aakeys:
            sigma = (aalist[i][2] + aalist[j][2]) / 10. / 2.
            nb.pair_coeff.set(i, j, lam=(aalist[i][3]+aalist[j][3])/2.,
                            epsilon=0.8368, sigma=sigma, r_cut=3*sigma)

    # Electrostatics
    yukawa = hoomd.md.pair.yukawa(r_cut=0.0, nlist=nl)
    for i,atom1 in enumerate(aakeys):
        for j,atom2 in enumerate(aakeys):
            yukawa.pair_coeff.set(atom1,atom2,epsilon=aalist[atom1][1]*aalist[atom2][1]*1.73136, kappa=1.0, r_cut=3.5) 
    # Group Particles
    all = hoomd.group.all()

    # Set up integrator
    hoomd.md.integrate.mode_standard(dt=0.01) # Time units in ps
    kTinput = T_sim * 8.3144598/1000.
    integrator = hoomd.md.integrate.langevin(group=all, kT=kTinput, seed=np.random.randint(1, 2**31))
    for cnt,i in enumerate(aakeys):
        integrator.set_gamma(i,gamma=aamass[cnt]/1000.0)

    # Output log file with box dimensions and restart file after box resizing
    hoomd.analyze.log(filename=f'{folder}.log', quantities=['potential_energy','kinetic_energy','temperature','pressure_xx','pressure_yy','pressure_zz','lx','ly','lz'], period=N_save, overwrite=True, header_prefix='#')
    hoomd.dump.gsd(f'{folder}.gsd', period=N_save, group=all, truncate=True)
    hoomd.dump.dcd(f'{folder}.dcd', period=N_save, group=all, overwrite=True)
    # Run resizing simulation
    hoomd.run(tsteps=N_steps)

seq = {'R':'ARG','H':'HIS','K':'LYS','D':'ASP','E':'GLU',
       'S':'SER','T':'THR','N':'ASN','Q':'GLN','C':'CYS',
       'U':'SEC','G':'GLY','P':'PRO','A':'ALA','V':'VAL',
       'I':'ILE','L':'LEU','M':'MET','F':'PHE','Y':'TYR',
       'W':'TRP'}

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--replica',nargs='?',const='', type=int, default=10)
args = parser.parse_args()

protein = args.name
cal2j = 4.184 # 1 cal = 4.184 J

folder = f'traj/{protein}/hps/1chain'
if not os.path.exists(folder):
    os.makedirs(folder)

ff_para = '../data/stats_module.dat'
aalist = {}
with open(ff_para, 'r') as fid:
    for i in fid:
        if i[0] != '#':
            tmp = i.rsplit()
            aalist[tmp[0]] = np.loadtxt(tmp[1:], dtype=float)
aakeys = list(aalist.keys())

proteins = pd.read_pickle('../data/allproteins.pkl').astype(object)
prot = proteins.loc[protein]
fasta = prot.fasta
T_sim = prot.temp

chain_id = []
chain_mass = []
chain_charge = []
for i in range(len(fasta)):
    one = fasta[i]
    if one in seq:
        iname = seq[one]
        chain_id.append(aakeys.index(iname))
        chain_mass.append(aalist[iname][0])
        chain_charge.append(aalist[iname][1])

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

s = gsd.hoomd.Snapshot()
s.particles.N = chain_length
s.particles.types = aakeys
s.particles.typeid = chain_id
s.particles.mass = chain_mass
s.particles.charge = chain_charge
# bond
pos = []
for i in range(len(chain_id)):
    pos.append((0, 0, (i-int(len(chain_id)/2))*bond_length))
pos = np.array(pos)
s.particles.position = pos
# position
nbonds = chain_length - 1
s.bonds.N = nbonds
s.bonds.types = ['AA_bond']
s.bonds.typeid = [0] * (nbonds)
bond_pairs = np.zeros((nbonds, 2), dtype=int)
for i in range(nbonds):
    bond_pairs[i, :] = np.array([i, i+1])
s.bonds.group = bond_pairs
# box size
s.configuration.dimensions = 3
s.configuration.box = [box_length, box_length, box_length, 0, 0, 0]
s.configuration.step = 0

startfile = f'{folder}/start.gsd'
f = gsd.hoomd.open(name=startfile, mode='wb')
f.append(s)
f.close()

time0 = time.time()
hoomd.context.initialize("")
for irep in range(args.replica):
    with hoomd.context.SimulationContext():
        run_single_chain_simulation(startfile, f'{folder}/replica_{irep}', aalist, aakeys, aamass, T_sim)
print(f'Simulations completed in {time.time()-time0:.2f} seconds.')