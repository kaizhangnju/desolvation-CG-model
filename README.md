# Effect of Desolvation on Biomolecular Liquid-liquid Phase Separation

This repository contains the simulation scripts, analysis utilities, and plotting routines used for the desolvation-parameterized coarse-grained modeling of intrinsically disordered proteins (IDPs).  
It provides a complete workflow to (i) perform single-chain and slab simulations, (ii) analyze temperature-dependent conformational and phase behavior, and (iii) reproduce all publication-quality figures.

## Overview

This repository represents the code component of the desolvation project and includes:

- Implementations of the desolvation-parameterized coarse-grained model
- Scripts for single-chain simulations to quantify temperature-dependent chain compaction
- Slab simulation workflows for probing phase separation behavior
- Residue parameter files for the HPS and CALVADOS models
- Protein sequence files and example `.dat` inputs for running simulations directly
- Plotting scripts used to generate all figures in the manuscript

## Repository Structure

### Simulation Scripts

All the simulation scripts are included in the `script/` directory.

Scripts beginning with `1chain_` perform **single-chain simulations**.  
Scripts beginning with `slab_` perform **slab simulations** across different temperatures.

The scripts resolve their input paths relative to the `script/` directory. Run the
following commands from that directory:

```bash
cd script
```

They require a compatible legacy HOOMD-blue 2.x environment with `azplugins`,
as well as the Python packages imported by the scripts (including NumPy, pandas,
MDTraj, and GSD). The slab examples run the production length specified in the
scripts (`1e8` steps); they are not quick smoke tests.

#### Single chain simulations
```bash
# with HPS model
python 1chain_HPS.py --name FUS --replica 10

# HPS + desolvation
python 1chain_HPSdes.py --name FUS --replica 10 --alpha_b 0.33 --alpha_w 0.06 --alpha_e 1.0

# CALVADOS model
python 1chain_CALVADOS.py --name FUS --replica 10

# CALVADOS2 + desolvation (manuscript parameterization)
python 1chain_CALVADOSdes.py --name FUS --replica 10 --alpha_b 0.30 --alpha_w 0.03 --alpha_e 1.31
```

#### Slab simulations
```bash
# HPS model
python slab_HPS.py --name FUS --temp 300

# HPS + desolvation
python slab_HPSdes.py --name FUS --temp 300 --alpha_b 0.33 --alpha_w 0.06 --eps_scale 1.0

# CALVADOS model
python slab_CALVADOS.py --name FUS --temp 300

# CALVADOS2 + desolvation (manuscript parameterization)
python slab_CALVADOSdes.py --name FUS --temp 300 --alpha_b 0.30 --alpha_w 0.03 --eps_scale 1.31
```

* `--name`: Protein identifier used to locate sequence files.
* `--temp`: Simulation temperature (slab simulations only).
* `--replica`: Number of independent replicas for single chain simulations. *Default value: 10*.
* `--alpha_b`: Ratio $\alpha_\mathrm{b} = \epsilon_\mathrm{b}/\epsilon$ defining the relative height of the desolvation barrier.
* `--alpha_w`: Ratio $\alpha_\mathrm{w} = \epsilon_\mathrm{w}/\epsilon$ defining the relative depth of the water-mediated well.
* `--alpha_e`: Global scaling factor applied to the base interaction energy $\epsilon$ in `1chain_*des.py`.
* `--eps_scale`: Equivalent global scaling factor used by `slab_*des.py`.

For the CALVADOS2-desolvation examples, `1.31` corresponds to the manuscript
energy scale of $0.262\,\mathrm{kcal\,mol^{-1}}$ relative to the scripts' base
energy scale of $0.2\,\mathrm{kcal\,mol^{-1}}$.

Users may generate their own `.dat` files following the format provided in `data/`.

### Plot Scripts

All plotting scripts required to reproduce the figures in the manuscript are stored in the plot/ directory.

Note: Several scripts assume specific relative paths to simulation outputs.
Modify paths as necessary before execution.

## Data Attribution & Citation

Some sequence data, experimental Rg values, and CALVADOS residue parameters originate from:

* Dignon et al., *PLoS Computational Biology*, 2018 ([HPS](https://doi.org/10.1371/journal.pcbi.1005941))

* Tesei G and Lindorff-Larsen K. 2022. *Improved Predictions of Phase Behaviour of IDPs by Tuning the Interaction Range* (Version 1.2) [Computer software]. Zenodo. [https://doi.org/10.5281/zenodo.7437501](https://doi.org/10.5281/zenodo.7437501)

* Additional references listed in the manuscript

Users should cite the corresponding papers when using these files.
