# MM_Promoter

Markovian Model for Gene Promoter Regulation.

This repository implements a Markovian model for simulating gene promoter dynamics, replacing traditional CME-based promoter binding/unbinding reactions with explicit Markovian state transitions. It includes implementations for several gene regulatory networks from the associated paper.

## Features

- **Markovian Dynamics**: Explicit tracking of promoter states (Empty, Active, Repressed).
- **Hybrid CME-ODE Simulation**: Integrated with Lattice Microbes (jLM) for spatial stochastic simulation.
- **Multiple Models**: Includes implementations for:
    - **Galactose Switch**: Detailed model of the GAL1, GAL2, GAL3, GAL80 network.
    - **Toggle Switch**: Bistable switch with mutual repression.
    - **Repressilator**: Synthetic three-gene oscillatory network.
    - **Goodwin Oscillator**: Negative feedback oscillator with mRNA and protein steps.
    - **I1-FFL**: Type 1 Incoherent Feed-Forward Loop (adaptive pulse generator).
    - **p53-Mdm2**: Oscillatory system involved in DNA damage response.
    - **NF-kappa B**: Signaling pathway involved in immune response.

## Repository Structure

```
MM_Promoter/
├── models/
│   ├── galactose/            # GAL system implementation
│   ├── toggle_switch/        # Toggle switch model
│   ├── repressilator/        # Repressilator model
│   ├── goodwin_oscillator/   # Goodwin oscillator model
│   ├── i1_ffl/               # I1-FFL model
│   ├── p53_mdm2/             # p53-Mdm2 model
│   └── nf_kappab/            # NF-kappa B model
├── examples/                 # General analysis notebooks
├── environment.yml           # Conda environment
└── LICENSE                   # MIT License
```

## Installation

Dependencies:
- Python >= 3.8
- `numpy`, `scipy`, `pandas`, `h5py`
- `latticemicrobes` (jLM)

See `environment.yml` for conda environment setup.

## Usage

Each model directory contains its own simulation scripts. For example, to run the Galactose model validation:

```bash
python models/galactose/run_cme_validation.py
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{wu2024markovian,
  title={Efficient Simulation of Gene Regulation Using Markovian Promoter Models},
  author={Wu, Tianyu and Roberts, Elijah},
  journal={Bioinformatics},
  year={2024},
  publisher={Oxford University Press}
}
```
