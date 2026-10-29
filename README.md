# Mathematical Modelling of Tumour Growth Inhibition

This repository contains the Python scripts developed for the numerical simulations and parameter estimation presented in the dissertation *Mathematical Modelling and Analysis of Tumour Growth Inhibition under Dasatinib and PEITC Treatment*.

The scripts implement a mathematical model describing tumour cell viability, FAK activity, and the effects of Dasatinib and PEITC, both individually and in combination.

## Repository contents

- `otimizacao_PD0_global.py` — Parameter estimation for the untreated control condition.
- `otimizacao_D_global.py` — Parameter estimation for the Dasatinib treatment.
- `otimizacao_P_global.py` — Parameter estimation for the PEITC treatment.
- `otimizacao_sinergia_global.py` — Estimation of the interaction parameter for the combined Dasatinib–PEITC treatment.
- `requirements.txt` — Python dependencies required to run the scripts.

## Requirements

- Python 3.9 or later
- NumPy
- pandas
- Matplotlib
- SciPy

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/tumour-growth-inhibition-model.git
cd tumour-growth-inhibition-model
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the scripts

The scripts should be executed in the following order:

```bash
python otimizacao_PD0_global.py
python otimizacao_D_global.py
python otimizacao_P_global.py
python otimizacao_sinergia_global.py
```

Each script automatically creates a dedicated output directory containing the corresponding plots and numerical results:

- `Results_PD0`
- `Results_D`
- `Results_P`
- `Results_synergy`

## Parameter estimation

The parameter estimation procedure combines global optimization using Differential Evolution with local refinement using nonlinear least squares (`scipy.optimize.least_squares`).

The scripts use experimental tumour cell viability data embedded in the source code. Parameter estimates obtained in earlier stages are used as fixed values in subsequent stages.

## Reproducibility

The random seed can be adjusted through the `Aleatorio` variable in each script.

The numerical results may vary depending on the optimization settings, random seed, and computational environment.
