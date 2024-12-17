# saxs_data_processing

This repository contains code to support automated analysis of small angle X-ray scattering data (SAXS). SAXS is a materials characterization technique that provides volume-averaged structural information. It is a powerful tool for understanding size, shape, and order of materials systems, but data analysis is a very manual, intuition-based process. We are interested in integrating SAXS into closed-loop autonomous experimentation systems for the optimization of materials properties (ie, a self driving lab). This requires automated capabilities In particular, this repository was developed with an eye for automating SAXS data processing for measurements of spherical silica nanoparticle samples, although it should be generalizable to other systems. This library is heavily used in our nanoparticle optimization campaigns, which can be found at [https://github.com/pozzo-research-group/silica-np-synthesis](https://github.com/pozzo-research-group/silica-np-synthesis)

## Supported tasks
This library assumes you are starting with reduced 3-column (q/I/sig) 1-dimensional data from a Xenocs Xeuss instrument. It has capabilities for:

- Background subracttion and vlid data range clipping
- q- range merging
- Convenience functions for SASview model fitting
- Target comparison calculation

## Installation

Use of a conda environment is recommended. Developed with python 3.12.

0. Create a conda environment if you don't already have one:
```
conda create -n sas python=3.12
```
```
conda activate sas
```

1. Clone this repository
```
git clone https://github.com/pozzo-research-group/saxs_data_processing
```

2. From the root of the repository, install with pip.

- First, install requirements.txt:
```
pip install -r requirements.txt
```
- Now, install this project:
```
pip install .
```


## Running examples
See the [usage_examples](./examples/usage_examples.ipynb) notebook for examples of how to use this library.
##
