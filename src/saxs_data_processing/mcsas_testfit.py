import h5py, sys, os
import numpy as np
import pandas
# import scipy
# import multiprocessing
from pathlib import Path

# load required modules
homedir = os.path.expanduser("~")
# disable OpenCL for multiprocessing on CPU
os.environ["SAS_OPENCL"] = "none"

# CHANGE to location where the SasView/sasmodels are installed
sasviewPath = os.path.join(homedir, "Code", "sasmodels")  # <-- change! 
if sasviewPath not in sys.path:
    sys.path.append(sasviewPath)
# import from this path
import sasmodels
import sasmodels.core
import sasmodels.direct_model

# CHANGE this one to whereever you have mcsas3 installed:
mcsasPath = os.path.join(homedir, "Code", "mcsas3")  # <-- change!
if mcsasPath not in sys.path:
    sys.path.append(mcsasPath)

# import from this path:
from mcsas3 import McHat
from mcsas3 import McData1D, McData2D
from mcsas3.mcmodelhistogrammer import McModelHistogrammer
from mcsas3.mcanalysis import McAnalysis
# optimizeScalingAndBackground: takes care of the calculation of the reduced chi-squared value, after a least-squares optimization for the scaling and background factors.
# McModel: extends the SasModel with information on the parameter set and methods for calculating a total scattering intensity from multiple contributions. It also tracks parameter bounds, random generators and picks.
# McOpt: contains mostly settings related to the optimization process. Also keeps track of the contribution to optimize.
# McCore: Contains the methods required to do the optimization. 

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("ggplot")



def mcsas_fit(fp_load, fp_write, n_threads = 40):
    # set a filename for documenting the fit:
    resPath = Path("mcsas_results", fp_write)
    # delete if it exists:
    if resPath.is_file(): resPath.unlink()

    mds = McData1D.McData1D(
        filename=Path("init_random_data", fp_load),
        nbins=100, # no rebinning in this example
        #dataRange = [0.01, 1], # this clips the data to the specified range

        # arguments for pandas.read_csv:
        csvargs = {"sep" : "\s+|\t+|\s+\t+|\t+\s+", # field delimiter, for flexible whitespace, use: "\s+|\t+|\s+\t+|\t+\s+" (https://stackoverflow.com/questions/15026698/how-to-make-separator-in-pandas-read-csv-more-flexible-wrt-whitespace-for-irreg#15026839)
                "skipinitialspace" : True, # ignore initial blank spaces
                "skip_blank_lines" : True, # ignore lines with nothing in them
                "skiprows" : 149, # skip this many rows before reading data (useful for PDH, which I think has five (?) header rows?)
                "engine": "python", # most flexible
                "header" : None, # let's not read any column names since they're unlikely to match with our expected column names:
                "names": ["Q", "I", "ISigma"], # our expected column names
                "index_col" : False}, # no index column before every row (who does this anyway?)
    )
    
    # store the data and all derivatives in the output file:
    mds.store(resPath)
    

    print(f'data fed to McSAS3 is {mds.measDataLink}')
    md = mds.measData.copy() # here we copy the data we want for fitting.

    mds.dataRange = [md['Q'][0].min(), md['Q'][0].max()]

    model = sasmodels.core.load_model_info('sphere')
    
    mh = McHat.McHat(
                modelName="sphere", # the model name chosen from the list above
                nContrib=300, # number of contributions, 300 normally suffice
                modelDType="default", # choose "fast" for single-precision calculations at your own risk
                fitParameterLimits={"radius": (20, 250)}, # this is the parameter we want to MC optimize within these bounds
                staticParameters={ # these are the parameters we don't want to change:
                    "background": 0, # is optimized separately, always set to zero
                    "scale": 1, # ibid.
                    "sld": 8.575, # SLD of silica
                    "sld_solvent": 9.611,# SLD of ethanol
                    },
                maxIter=100000, # don't try more than this many iterations
                convCrit=1, # convergence criterion, should be 1 if reasonable uncertainty estimates are provided, to prevent over- or under-fitting
                nRep=30, # number of independent repetitions of the optimization procedure. 10 for fast, 50-100 for publications
                nCores = n_threads, # number of threads to spread this over. Set to 0 for automatic detection of maximum number of threads
                seed=None, # random number generator seed. Set to a specific value for reproducible random numbers
            )
    mh.store(resPath)


    mh.run(md, resPath)

    return