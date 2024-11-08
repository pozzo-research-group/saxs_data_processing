import pandas as pd
from saxs_data_processing.utils import numerify, is_valid_uuid
import os
from sasdata.dataloader.loader import Loader
import numpy as np
import warnings
import glob
import re


def read_1D_data(fp):
    """Read a 1D Xenocs data .dat file and return data as a pandas dataframe with q/I/sig values and metadata with header data

    :param fp: Filepath to read
    :type fp: str
    :return data: q/I/sig data
    :rtype data: pandas.core.frame.DataFrame
    :return metadata: metadata read from file header
    :rtype metdata: dict
    """
    metadata = {}
    q = []
    I = []
    sig = []
    header_count = 0

    in_data = False  # flag for if we are read into data yet

    with open(fp, "rt") as f:
        for line in f:
            if not in_data:
                if line[:2] == "##":  # metadata section header
                    header_count += 1
                    continue
                elif line[:2] == "# ":
                    items = line.split()
                    if len(items) == 2:
                        items.append(None)
                    metadata[items[1]] = numerify(items[2])

                elif line.strip()[0] == "q" and header_count == 2:
                    in_data = True
            elif in_data:
                vals = line.split()
                q.append(numerify(vals[0]))
                I.append(numerify(vals[1]))
                sig.append(numerify(vals[2]))

    data = pd.DataFrame({"q": q, "I": I, "sig": sig})

    return data, metadata


def write_dat(data, metadata, fp):
    """Write a .dat datafile for data and metadata at fp

    :param data: q/I/sig data to write
    :type data: pandas.core.frame.DataFrame
    :param metadata: metadata dictionary with metadata to write to file header
    :type metedata: dict
    :param fp: Filepath
    :type fp: str
    """

    with open(fp, "wt") as f:
        f.write("#" * 80 + "\n")
        for key, val in metadata.items():
            f.write("# " + str(key) + " " * (32 - len(str(key))) + str(val) + "\n")
        f.write("#" * 80 + "\n")
        f.write("q(A-1)                    I(q)                      Sig(q) \n")
        for i, row in data.iterrows():
            q = str(row["q"])
            i = str(row["I"])
            sig = str(row["sig"])
            f.write(q + " " * (26 - len(q)) + i + " " * (26 - len(i)) + sig + "\n")


def sasdata_to_df(data):
    """
    Convert a sasdata data object into a dataframe compatible with Brenden's processing tools

    Curently does not convert metadata, just q/I/sig
    :param data: sasdata data object
    :type data: sasdata
    :return df: 3-column dataframe with columns 'q', 'I', 'sig'
    """

    return pd.DataFrame({"q": data.x, "I": data.y, "sig": data.dy})


def df_to_sasdata(df):
    """
    Convert a brenden format q/I/sig dataframe to sasdata object

    Currently a horrendous kludge - writes to disk then re-loads.
    """
    fp = "temp.dat"
    write_dat(df, {"metadata": None}, fp)
    loader = Loader()
    data = loader.load(fp)
    os.remove(fp)

    data = data[0]
    data.qmin = data.x.min()
    data.qmax = data.x.max()
    data.mask = np.isnan(data.y)

    return data


def load_data_files_biocube(root_dirs, bio_conf):
    """
    Finds all .dat files, returns those with bio configuration equal to bio_conf

    Loads Xenocs .dat files measured with biocube into Brenden's df format. Non-recursively checks all root directories for .dat files, parses them, if biocube conf is correct, returns

    :param root_dirs: List of directories to look in for .dat files
    :type root_dirs: list of strings
    :param bio_conf: Biocube configuration number (SAXS config ex ESAXS) to load. ESAXS is 24
    :type bio_conf: int
    :return sample_data: dictionary of sample_uuid_val:(brenden df, metadata) format
    :type sample_data: dict
    :return data_fps: dictionary of sample_uuid_val:filepath
    :type data_fps: dict
    :return fps_agg: list of all filepaths loaded
    :type fps_agg: list
    """

    # 1. Find all the filepaths in root directories of .dat files
    fps_agg = []
    for root_dir in root_dirs:
        fps = glob.glob("**.dat", root_dir=root_dir, recursive=False)
        fps_agg.extend([root_dir + fp for fp in fps])

    # 2. process filepaths, get uuids for those that are sample data (as opposed to backgrounds). load those that match target config
    data_fps = {}
    sample_data = {}
    not_data_count = 0
    not_esaxs_count = 0
    for fp in fps_agg:
        uuid_str = re.split("_+", fp.split("/")[-1])[2]

        if is_valid_uuid(uuid_str):
            data = io.read_1D_data(fp)
            if data[1]["BIO_SYSTEM_CONF"] == bio_conf:
                try:
                    sample_data[uuid_str]
                    warnings.warn(
                        f"Duplicate UUID found for {uuid_str}. Check your file naming"
                    )
                except KeyError:
                    sample_data[uuid_str] = data
                data_fps[uuid_str] = fp
            else:
                not_esaxs_count += 1
        else:
            not_data_count += 1

    return sample_data, data_fps, fps_agg
