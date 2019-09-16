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
        'files/raw/processed_single_cell_data.csv',
        memory_map=True, index_col=0, header=None, skiprows=1).T
    selected_pandas = all_pandas[column_array]
    all_array = selected_pandas.to_numpy(dtype='float32', copy=True)
    all_array = all_array.transpose()
    print(all_array.shape)  # (6189, 23946)

    # Importing tomoSeq data and converting to numpy array:
    tomo_array = pandas.read_csv(
        'files/raw/embryo_average_landmark_genes_landmarkEbar.csv')
    tomo_array = tomo_array.loc[:, '1':]
    tomo_array = tomo_array.to_numpy(dtype='float32', copy=True)

    # Saving to file:
    if not os.path.exists('files/'):
        os.mkdir('files/')
    np.save('files/48seqcounts.npy', sc_array)
    np.save('files/avgtomodata.npy', tomo_array)
    np.save('files/allseqcounts.npy', all_array)


if __name__ == '__main__':
    main()
