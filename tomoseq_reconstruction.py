"""
A script to reconstruct a pseudo-tomoseq dataset from a) calculated RCC values
and b) pseudo-RCC values from the neural net.

Example:
python tomoseq_reconstruction.py
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def main():
    """
    Import data, assign each cell to a bin, calculate resulting
    RNAseq signatures for each bin and compare to
    avgTomoSeq data.
    """

    # Import data:
    sc_dataset = np.load('files/48seqcounts.npy')
    RCC = np.load('files/RCClandmarkvalues.npy')
    tomoseq_dataset = np.load('files/fiftybintomoseq.npy')
    pseudoRCC = np.load(
        'files/predictions/dist_predictions_2019-09-23 14:40.npy')

    # Generating assignment matrix:
    RCC_assignment = RCC.argmax(axis=1)
    pseudoRCC_assignment = pseudoRCC.argmax(axis=1)

    # Assigning counts from each matrix:
    RCC_assembled = np.zeros((48, 50))
    pseudoRCC_assembled = np.zeros((48, 50))

    for i in range(50):
        cell_indices = np.nonzero(RCC_assignment == i)
        binned_cells = sc_dataset[cell_indices, :]
        if binned_cells.shape[1] == 0:
            continue
        RCC_assembled[:, i] = np.mean(binned_cells, axis=1)
    for i in range(50):
        cell_indices = np.nonzero(pseudoRCC_assignment == i)
        binned_cells = sc_dataset[cell_indices, :]
        if binned_cells.shape[1] == 0:
            continue
        pseudoRCC_assembled[:, i] = np.mean(binned_cells, axis=1)

    # Calculating scores:
    RCCcorr_array = []
    pseudocorr_array = []
    for i in range(48):
        correlation, _ = spearmanr(RCC_assembled[i, :], tomoseq_dataset[i, :])
        RCCcorr_array.append(correlation)
    for i in range(48):
        correlation, _ = spearmanr(
            pseudoRCC_assembled[i, :], tomoseq_dataset[i, :]
            )
        if np.isnan(correlation):
            continue
        pseudocorr_array.append(correlation)

    RCCscore = np.median(np.asarray(RCCcorr_array))
    pseudoRCCscore = np.median(np.asarray(pseudocorr_array))
    print('RCC score: ' + str(RCCscore))
    print('PseudoRCC score: ' + str(pseudoRCCscore))

    # Plotting values:
    if not os.path.exists('files/reconstruction_images/'):
        os.mkdir('files/reconstruction_images/')
    if not os.path.exists('files/reconstruction_plots/'):
        os.mkdir('files/reconstruction_plots/')

    for i in range(48):
        fig, axs = plt.subplots(3, sharex=True, figsize=(8, 4))
        fig.suptitle('Distributions of gene')
        axs[0].plot(tomoseq_dataset[i, :])
        axs[1].plot(RCC_assembled[i, :])
        axs[2].plot(pseudoRCC_assembled[i, :])
        axs[2].set(xlabel='A/P axis')
        fig.text(0.04, 0.5, 'Arbitrary expression',
                 va='center', rotation='vertical')
        plt.savefig('files/reconstruction_plots/gene' + str(i))
        plt.close()

    _ = sns.heatmap(
        tomoseq_dataset-tomoseq_dataset.mean(axis=1, keepdims=True))
    plt.savefig('files/reconstruction_images/tomoseq_heatmap.png',
                dpi=300, orientation='portrait')
    plt.clf()

    _ = sns.heatmap(
        RCC_assembled-RCC_assembled.mean(axis=1, keepdims=True))
    plt.savefig('files/reconstruction_images/RCC_heatmap.png',
                dpi=300, orientation='portrait')
    plt.clf()

    _ = sns.heatmap(
        pseudoRCC_assembled-pseudoRCC_assembled.mean(axis=1, keepdims=True))
    plt.savefig('files/reconstruction_images/pseudoRCC_heatmap.png',
                dpi=300, orientation='portrait')
    plt.clf()


if __name__ == '__main__':
    main()
