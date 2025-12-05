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

#### Single chain simulations
```bash
# with HPS model
python 1chain_HPS.py --name FUS --replica 10

# HPS + desolvation
python 1chain_HPSdes.py --name FUS --replica 10 --alpha_b 0.33 -alpha_w 0.06 -alpha_e 1.0

# CALVADOS model
python 1chain_CALVADOS.py --name FUS --replica 10

# CALVADOS + desolvation
python 1chain_CALVADOSdes.py --name FUS --replica 10 --alpha_b 0.33 -alpha_w 0.06 -alpha_e 1.0
```

#### Slab simulations
```bash
# HPS model
python slab_HPS.py --name FUS --temp 300

# HPS + desolvation
python slab_HPSdes.py --name FUS --temp 300 --alpha_b 0.33 -alpha_w 0.06 -alpha_e 1.0

# CALVADOS model
python slab_CALVADOS.py --name FUS --temp 300

# CALVADOS + desolvation
python slab_CALVADOSdes.py --name FUS --temp 300 --alpha_b 0.33 -alpha_w 0.06 -alpha_e 1.0
```

* `--name`: Protein identifier used to locate sequence files.
* `--temp`: Simulation temperature (slab simulations only).
* `--replica`: Number of independent replicas for single chain simulations. *Default value: 10*.
* `--alpha_b`: Ratio $\alpha_\mathrm{b} = \epsilon_\mathrm{b}/\epsilon$ defining the relative height of the desolvation barrier.
* `--alpha_w`: Ratio $\alpha_\mathrm{w} = \epsilon_\mathrm{w}/\epsilon$ defining the relative depth of the water-mediated well.
* `--alpha_e`: Global scaling factor applied to the base interaction energy $\epsilon$.

Users may generate their own `.dat` files following the format provided in `data/`.

### Plot Scripts

All plotting scripts required to reproduce the figures in the manuscript are stored in the plot/ directory.

Note: Several scripts assume specific relative paths to simulation outputs.
Modify paths as necessary before execution.

## Data Attribution & Citation

Some sequence data, experimental Rg values, and CALVADOS residue parameters originate from:

* Dignon et al., *PLoS Computational Biology*, 2018 ([HPS](https://doi.org/10.1371/journal.pcbi.1005941))

* Tesei et al., *PNAS*, 2021 ([CALVADOS]((https://doi.org/10.1073/pnas.2111696118)))

* Additional references listed in the manuscript

Users should cite the corresponding papers when using these files.