"""
Convert scRNAseq, tomoseq and RCC 48 data into numpy array,
and save locally to files.

Example:
python import_data.py
"""
import os
import pandas
import numpy as np


def main():
    """
    Convert to numpy array and save file.
    """

    # Importing landmark gene scRNAseq data and converting to numpy array:
    sc_pandas = pandas.read_csv(
        'files/raw/scaled_single_cells_and_landmarkgenes_expc.csv')
    sc_pandas = sc_pandas.loc[:, 'cells.DEW038_TGAACGCTCAG_AACCCTTG':]
    column_array = sc_pandas.columns.tolist()
    sc_array = sc_pandas.to_numpy(dtype='float32', copy=True)
    sc_array = sc_array.transpose()
    print(sc_array.shape)  # (6189, 48)

    # Importing all gene scRNAseq data and converting to numpy array:
    for i in range(len(column_array)):
        column_array[i] = column_array[i][6:]
    all_pandas = pandas.read_csv(
        'files/raw/GSM3067194_18hpf_nm.csv',
        memory_map=True)
    all_pandas = all_pandas.set_index('Row')
    all_pandas = all_pandas[column_array]
    all_array = all_pandas.to_numpy(dtype='float32', copy=True)
    all_array = np.transpose(all_array)
    print(all_array.shape)  # (6189, 23946)

    # Importing landmark tomoSeq data and converting to numpy array:
    tomo_array = pandas.read_csv(
        'files/raw/embryo_average_landmark_genes_landmarkEbar.csv')
    tomo_array = tomo_array.loc[:, '1':]
    tomo_array = tomo_array.to_numpy(dtype='float32', copy=True)

    # Importing sym to ensembl lookup table:
    sym_ensembl = pandas.read_csv(
        'files/raw/varGenes_symtoEnsembl.csv').iloc[:, 1:]
    sym_ensembl.columns = sym_ensembl.columns.str.strip(
        ).str.lower().str.replace(' ', '_')
    sym_ensembl.dropna(inplace=True)
    sym_ensembl.to_csv(path_or_buf='files/sym_ensembl.csv')

    # Generating list of shared genes:
    all_tomo = pandas.read_csv('files/raw/tomo_average.eucl.csv', index_col=0)
    tomo_list = list(all_tomo.index.values)
    sym_ensembl = sym_ensembl.set_index(
        'ensemblid', drop=False).loc[tomo_list, :]
    sym_ensembl.dropna(inplace=True)

    # Extracting shared genes
    var_list = sym_ensembl.loc[:, 'ensemblid'].values.tolist()
    var_tomo = all_tomo.loc[var_list, :]
    var_tomo.sort_index(inplace=True)
    var_tomo.to_csv(path_or_buf='files/vartomo.csv')
    var_tomo = var_tomo.to_numpy(dtype='float32', copy=True)

    # Importing spatially variable list of genes into pandas:
    var_list = sym_ensembl.loc[:, 'genesymbol'].values.tolist()[0:500]
    var_seq = all_pandas.loc[var_list, :]
    lookup_dict = sym_ensembl.set_index('genesymbol').T.to_dict(
        orient='list')
    print(lookup_dict)
    var_seq.rename(index=lookup_dict, inplace=True)
    var_seq.sort_index(inplace=True, axis=0)
    var_seq.to_csv(path_or_buf='files/varseq.csv')
    var_seq = var_seq.to_numpy(dtype='float32', copy=True)
    var_seq = np.transpose(var_seq)
    print(var_seq.shape)

    # Saving to file:
    if not os.path.exists('files/'):
        os.mkdir('files/')
    np.save('files/48seqcounts.npy', sc_array)
    np.save('files/avgtomodata.npy', tomo_array)
    np.save('files/allseqcounts.npy', all_array)
    np.save('files/variable_dataset.npy', var_seq)
    np.save('files/variable_tomo.npy', var_tomo)


if __name__ == '__main__':
    main()
