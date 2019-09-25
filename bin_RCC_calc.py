"""
A script to bin the avgTomoSeq data into 50 sections,
and to calculate the RCCs of the entire single cell
!scaled! dataset for training of our first proper model.

Example:
python bin_RCC_calc.py
"""
import numpy as np
from scipy.stats import spearmanr


def main():
    """
    Loading data, generating bins and calculating RCC values.
    """

    # Load data:
    sc_data = np.load('files/48seqcounts.npy')
    print('Single cell data shape:' + str(sc_data.shape))
    tomo_data = np.load('files/avgtomodata.npy')
    print('Tomo data shape:' + str(tomo_data.shape))

    # Binning tomoseqdata:
    fifty_section_data = []
    for i in range(50):
        k = i*2
        binned_instance = np.sum(tomo_data[:, k:k+2], axis=1)
        fifty_section_data.append(binned_instance)
    fifty_section_data = np.stack(fifty_section_data)
    fifty_section_data = np.transpose(fifty_section_data)
    print('Fifty section data:' + str(fifty_section_data.shape))

    # Calculating of RCC for single cell data:
    rcc_matrix = []
    print('Calculating RCC values for sc dataset:')
    for i in range(sc_data.shape[0]):
        sc_instance = np.squeeze(sc_data[i, :])
        rcc_instance = []
        for j in range(100):
            rcc, _ = spearmanr(sc_instance, tomo_data[:, j])
            rcc_instance.append(rcc)
        rcc_instance = np.asarray(rcc_instance)
        rcc_matrix.append(rcc_instance)
        if i % 1000 == 0:
            print('>>>> ' + str(i) + ' cells processed.')
    rcc_matrix = np.stack(rcc_matrix)
    print(rcc_matrix.shape)

    # Saving data to file:
    np.save('files/fiftybintomoseq.npy', fifty_section_data)
    np.save('files/RCClandmarkvalues.npy', rcc_matrix)


if __name__ == '__main__':
    main()
