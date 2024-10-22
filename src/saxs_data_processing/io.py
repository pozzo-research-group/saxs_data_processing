import pandas as pd
from saxs_data_processing.utils import numerify


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

    with open(fp, 'rt') as f:
        for line in f:
            if not in_data:
                if line[:2] == '##':  #metadata section header
                    header_count += 1
                    continue
                elif line[:2] == '# ':
                    items = line.split()
                    if len(items) == 2:
                        items.append(None)
                    metadata[items[1]] = numerify(items[2])

                elif line.strip()[0] == 'q' and header_count == 2:
                    in_data = True
            elif in_data:
                vals = line.split()
                q.append(numerify(vals[0]))
                I.append(numerify(vals[1]))
                sig.append(numerify(vals[2]))

    data = pd.DataFrame({'q': q, 'I': I, 'sig': sig})

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

    with open(fp, 'wt') as f:
        f.write('#' * 80 + '\n')
        for key, val in metadata.items():
            f.write('# ' + str(key) + ' ' * (32 - len(str(key))) + str(val) +
                    '\n')
        f.write('#' * 80 + '\n')
        f.write(
            'q(A-1)                    I(q)                      Sig(q) \n')
        for i, row in data.iterrows():
            q = str(row['q'])
            i = str(row['I'])
            sig = str(row['sig'])
            f.write(q + ' ' * (26 - len(q)) + i + ' ' * (26 - len(i)) + sig +
                    '\n')
