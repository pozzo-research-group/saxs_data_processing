# saxs_data_processing

This repository contains code to support automated analysis of small angle X-ray scattering data (SAXS). SAXS is a materials characterization technique that provides volume-averaged structural information. It is a powerful tool for understanding size, shape, and order of materials systems, but data analysis is a very manual, intuition-based process. We are interested in integrating SAXS into closed-loop autonomous experimentation systems for the optimization of materials properties (ie, a self driving lab). This requires automated capabilities In particular, this repository was developed with an eye for automating SAXS data processing for measurements of spherical silica nanoparticle samples, although it should be generalizable to other systems.

## Supported tasks

- Background subract
- q- range merging
- Valid data range clipping
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

First, install requirements.txt:
```
pip install -r requirements.txt
```
Now, install this project:
```
pip install .
```


## Running examples
coming soon
##
