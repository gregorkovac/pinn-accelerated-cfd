<h1 align="center">
<br>
Accelerating numerical methods for solving Navier-Stokes equations with physics-informed machine learning
<br>
</h1>

My master's thesis!

## Abstract

The Navier-Stokes equations are a standard model for describing fluid behavior and are typically solved with computationally expensive numerical methods. We take the projection method, implement it, and identify solving the pressure-Poisson equation as the most computationally expensive step. We propose a hybrid method that replaces the pressure-Poisson solver with a physics-informed convolutional neural network. As a baseline, we train another physics-informed model for solving the Navier-Stokes equations directly. Both models are trained on two-dimensional stationary laminar flow cases using exclusively equation residuals, without ground-truth velocity and pressure fields. On the testing set, the baseline model achieves relative velocity errors in the $L_2$ norm of $0.23 \pm 0.05$, and our proposed method achieves errors of $0.06 \pm 0.03$. Furthermore, our method computes solutions almost three times faster than the numerical method, demonstrating the potential of hybrid methods to accelerate computational fluid dynamics.

## Project structure

- `data/` - flow data case parameters
- `models/` - output folder for the machine learning models
- `src/` - source code of the project
  - `analysis/` - code for method comparison/analysis
  - `data-generation/` - code for generating the data sets with the numeric method
  - `modeling/` - code for defining and training the machine learning models
  - `numeric-method` - code for the numeric method

## Requirements

Install all of the requirements for the project by running

```
pip install -r requirements.txt
```

## Running the numeric method

Run the numeric method by moving into the numeric method folder and running `main.py`. The method parameters and the simulated case can be adjusted inside of `main.py`.

```sh
cd src/numeric_method
python main.py

```

## Data generation

Generate the data with `src/data-generation/generate-data.py`. All of the possible parameters can be seen by running

```
python .\generate-data.py --help
```

## Training the models

Train the models with `src/modeling/train.py`. All of the possible parameters can be seen by running

```
python .\train.py --help
```
