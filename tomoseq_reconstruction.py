"""
A script to reconstruct a pseudo-tomoseq dataset from a) calculated RCC values and
b) pseudo-RCC values from the neural net.

Example:
python tomoseq_reconstruction.py
"""
#pylint: disable=import-error
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from scipy.stats import spearmanr

def main():
    """
    Import data, assign each cell to a bin, calculate resulting
    RNAseq signatures for each bin and compare to 
    avgTomoSeq data.
    """

    # Import data:
    sc_dataset = np.load('files/normalised48counts.npy')
    all_dataset = np.load('files/allseqcounts.npy')
    RCC = np.load('files/RCClandmarkvalues.npy')
    tomoseq_dataset = np.load('files/fiftybintomoseq.npy')

    # Generating model pseudo RCC dataset:
    model = load_model('files/models/2019-09-11 17:06.h5')
    pseudoRCC = model.predict(all_dataset)

    # Generating assignment matrix:
    RCC_assignment = RCC.argmax(axis=1)
    pseudoRCC_assignment = pseudoRCC.argmax(axis=1)

    # Assigning counts from each matrix:
    RCC_assembled = np.zeros((48, 50))        
    pseudoRCC_assembled = np.zeros((48, 50))

    for i in range(50):
        cell_indices = np.nonzero(RCC_assignment == i)
        binned_cells = sc_dataset[cell_indices, :]
        RCC_assembled[:, i] = np.mean(binned_cells, axis=1)
    for i in range(50):
        cell_indices = np.nonzero(pseudoRCC_assignment == i)
        binned_cells = sc_dataset[cell_indices, :]
        pseudoRCC_assembled[:, i] = np.mean(binned_cells, axis=1)

    # Calculating scores:
    RCCcorr_array = []
    pseudocorr_array = []
    for i in range(50):
        correlation, _ = spearmanr(RCC_assembled[:, i], tomoseq_dataset[:, i])
        RCCcorr_array.append(correlation)
    for i in range(50):
        correlation, _ = spearmanr(pseudoRCC_assembled[:, i], tomoseq_dataset[:, i])
        if np.isnan(correlation):
            continue
        pseudocorr_array.append(correlation)

    RCCscore = np.mean(np.asarray(RCCcorr_array))
    pseudoRCCscore = np.median(np.asarray(pseudocorr_array))
    print('RCC score: ' + str(RCCscore))
    print('PseudoRCC score: ' + str(pseudoRCCscore))

    # Plotting values:
    if not os.path.exists('files/reconstruction_images/'):
        os.mkdir('files/reconstruction_images/')

    _ = sns.heatmap(tomoseq_dataset)
    plt.savefig('files/reconstruction_images/tomoseq_heatmap.png', dpi=300, orientation='portrait')
    plt.show()
    plt.clf()

    _ = sns.heatmap(RCC_assembled)
    plt.savefig('files/reconstruction_images/RCC_heatmap.png', dpi=300, orientation='portrait')
    plt.show()
    plt.clf()

    _ = sns.heatmap(pseudoRCC_assembled)
    plt.savefig('files/reconstruction_images/pseudoRCC_heatmap.png', dpi=300, orientation='portrait')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()