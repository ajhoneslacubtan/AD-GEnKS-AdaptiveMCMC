AD-GEnKS-AdaptiveMCMC

A Python implementation of an Adaptive Metropolis-within-Gibbs sampler with Ensemble Kalman Smoother for spatiotemporal modeling.

## Installation

**Requirements:**
- Python 3.12+
- Julia 1.11.4+

1. First, install Julia requirements through Python REPL:
```python
python
>>> import julia
>>> julia.install()
>>> exit()
```

2. Then install Python dependencies:
```sh
pip install -r requirements.txt
```

## Overview

This project implements a Bayesian inference framework that combines:
- Gibbs sampling
- Adaptive Metropolis-Hastings
- Ensemble Kalman Smoother (EnKS)

For analyzing spatiotemporal data, particularly clear-sky index (CSI) and solar radiation data.

## Usage

For simulated data:
```sh
python main.py
```

For real data analysis:
```sh
python run_real_data.py
python run_real_data_csi.py
```

To run in background with logging:
```sh
./run_background.sh run_real_data.py
```

## Project Structure

- main.py - Simulation script
- run_real_data.py - Real data analysis for GHI
- run_real_data_csi.py - Real data analysis for CSI
- sampler - Core sampling algorithms
- data - Input data files (.nc and .tif formats)
- visualize.py - Visualization utilities

## Dependencies

Main dependencies:
- Julia
- NumPy
- xarray
- zarr
- arviz
- matplotlib
- pandas
- scipy

## License

MIT License

Copyright (c) 2025 AJ Jhones S. Lacubtan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
