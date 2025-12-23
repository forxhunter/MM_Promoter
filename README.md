# MM_Promoter

Markovian Model for Gene Promoter Regulation.

This repository implements a Markovian model for simulating gene promoter dynamics, focusing on the GAL regulatory network in yeast. It replaces traditional CME-based promoter binding/unbinding reactions with explicit Markovian state transitions.

## Features

- **Markovian Dynamics**: Explicit tracking of promoter states (Empty, Active, Repressed).
- **Hybrid CME-ODE Simulation**: Integred with Lattice Microbes (jLM) for spatial stochastic simulation.
- **Galactose Regulation**: Models the GAL1, GAL2, GAL3, GAL80, and reporter gene network.

## Installation

Dependencies:
- Python >= 3.8
- `numpy`, `scipy`, `pandas`, `h5py`
- `latticemicrobes` (jLM)

See `environment.yml` for conda environment setup.

## Usage

Examples are provided in the `examples/` directory.

```bash
# General simulation run
python src/run_cme_validation.py
```
