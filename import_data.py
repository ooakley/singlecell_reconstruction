"""
Convert scRNAseq, tomoseq and RCC 48 data into numpy array, 
and save locally to files.

Example:
python import_data.py
"""
import os
import random
import argparse
import pandas
import numpy as np

def main():
    """
    Convert to numpy array and save file.
    """

    # Importing scRNAseq data and converting to numpy array:
    sc_array = pandas.read_csv('files/raw/scaled_single_cells_and_landmarkgenes_expc.csv')
    sc_array = sc_array.loc[:, 'cells.DEW038_TGAACGCTCAG_AACCCTTG':]
    sc_array = sc_array.to_numpy(dtype='float32', copy=True)
    sc_array = sc_array.transpose()
    print(sc_array.shape)

    # Importing tomoSeq data and converting to numpy array:
    tomo_array = pandas.read_csv('files/raw/embryo_average_landmark_genes_landmarkEbar.csv')
    tomo_array = tomo_array.loc[:, '1':]
    tomo_array = tomo_array.to_numpy(dtype='float32', copy=True)

    # Saving to file:
    if not os.path.exists('files/'):
        os.mkdir('files/')
    np.save('files/normalised48counts.npy', sc_array)
    np.save('files/avgtomodata.npy', tomo_array)

if __name__ == '__main__':
    main()
