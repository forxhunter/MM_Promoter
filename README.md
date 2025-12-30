# MM_Promoter: Markovian Promoter Models

**A Mechanistic Alternative to Hill Functions in Gene Regulatory Networks**

This repository contains the reference implementation for the paper **"Markovian Promoter Models: A Mechanistic Alternative to Hill Functions in Gene Regulatory Networks"** (Wu, 2025). 

It implements a hybrid simulation framework that replaces phenomenological Hill functions with explicit Markovian transitions between discrete promoter states (e.g., Empty, Active, Repressed). This approach naturally captures ultrasensitivity, bursty expression, and combinatorial logic while maintaining computational efficiency (typically 10-20x faster than full SSA).

## Repository Structure

```
MM_Promoter/
├── core/                     # Core library
│   ├── markovian_promoter.py # Main MarkovianPromoterModel class
│   ├── multi_input_promoter.py # Multi-input promoter logic (AND/OR/NOT)
│   └── validation_utils.py   # Statistical validation tools
├── models/                   # Reference implementations
│   ├── galactose/            # Yeast GAL system (Main Case Study)
│   ├── toggle_switch/        # Bistable Toggle Switch
│   ├── repressilator/        # Synthetic Repressilator
│   ├── goodwin_oscillator/   # Goodwin Oscillator
│   ├── i1_ffl/               # I1-FFL Pulse Generator
│   ├── p53_mdm2/             # p53-Mdm2 Damped Oscillator
│   └── nf_kappab/            # NF-kappaB Signaling Pathway
├── scripts/                  # Figure reproduction scripts
├── examples/                 # Jupyter notebooks for easy start
├── environment.yml           # Conda environment definition
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/forxhunter/MM_Promoter.git
   cd MM_Promoter
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate mm_promoter
   ```

## Reproducing Results

We provide a master script to reproduce all figures from the manuscript.

```bash
# Generate all figures
python scripts/run_all_figures.py
```

Generated figures will be saved in the `figures/` directory.

### Individual Figure Reproduction

Each figure can also be generated individually:

**Figure 1: GAL System Dose-Response**
```bash
python scripts/figure_gal_dose_response.py
```

**Figure 2: GAL System Dynamics**
```bash
python scripts/figure_gal_timeseries.py
```

**Figure 3: Promoter State Dynamics**
```bash
# First regenerate data (if needed)
python scripts/hybrid_gal_simulator.py --mode promoter
# Plot
python scripts/figure_gal_promoter_states.py
```

**Figure 4: Simple Regulatory Motifs**
(Repressilator, Goodwin Oscillator, Toggle Switch)
```bash
python scripts/generate_combined_simple.py
```

**Figure 5: Advanced Regulatory Networks**
(I1-FFL, p53-Mdm2, NF-kappaB)
```bash
python scripts/generate_advanced_plots.py
```

## Model Descriptions

### 1. Markovian Hybrid Framework
The core innovation is replacing the quasi-equilibrium assumption ($P_{active} \approx [TF]^n / (K^n + [TF]^n)$) with a dynamic Markov chain for the promoter state $S$:
$$ \frac{dS}{dt} = \mathbf{Q}([TF]) \cdot S $$
This is coupled to ODEs for mRNA ($m$) and protein ($p$):
$$ \frac{dm}{dt} = \alpha(S) - \gamma_m m $$
$$ \frac{dp}{dt} = k_{tl} m - \gamma_p p $$

### 2. Yeast GAL System
- **Genes**: GAL1, GAL2, GAL3, GAL80, Reporter
- **Logic**: Gal4p activates, Gal80p represses by binding Gal4p
- **Features**: Ultrasensitivity from multi-site binding (GAL1: 4 sites), feedback loops
- **Location**: `models/galactose/`

### 3. Synthetic Circuits
- **Repressilator**: 3 genes (TetR, cI, LacI) in a repressible ring. Shows sustained oscillations.
- **Toggle Switch**: 2 mutually repressing genes. Shows bistability and memory.
- **Goodwin Oscillator**: Negative feedback loop with high cooperativity ($n=10$).
- **Location**: `models/repressilator/`, `models/toggle_switch/`, `models/goodwin_oscillator/`

### 4. Advanced Pathways
- **I1-FFL**: Incoherent Feed-Forward Loop. Generates pulses (adaptive response) to step inputs.
- **p53-Mdm2**: DNA damage response. Shows damped oscillations.
- **NF-kappaB**: Immune signaling. Involves nuclear-cytoplasmic shuttling.
- **Location**: `models/i1_ffl/`, `models/p53_mdm2/`, `models/nf_kappab/`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wu2025markovian,
  title={Markovian Promoter Models: A Mechanistic Alternative to Hill Functions in Gene Regulatory Networks},
  author={Wu, Tianyu},
  journal={arXiv preprint arXiv:2512.18442},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
