"""
A script to normalise all necessary data for the training of a neural network.

Example:
python normalise_data.py
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def apply_normparameters(dataset, mean, std, max, min):
    normalised_dataset = (dataset - mean)/std
    normalised_dataset = (normalised_dataset - min)/(max - min)
    return normalised_dataset


def calculate_normparameters(dataset):
    mean = np.mean(dataset, axis=0)
    mean = np.expand_dims(mean, axis=0)
    print(mean.shape)
    std = np.std(dataset, axis=0)
    std = np.expand_dims(std, axis=0)
    print(std.shape)
    normalised_dataset = (dataset - mean)/std
    max = np.max(normalised_dataset, axis=0)
    max = np.expand_dims(max, axis=0)
    min = np.min(normalised_dataset, axis=0)
    min = np.expand_dims(min, axis=0)
    return mean, std, max, min


def main():
    # Importing data:
    landmark_seq = np.load('files/48seqcounts.npy')
    print('lmseq shape: ' + str(landmark_seq.shape))
    all_seq = np.load('files/allseqcounts.npy')
    print('all_seq shape: ' + str(all_seq.shape))

    # Normalising landmark data:
    params = calculate_normparameters(landmark_seq)
    norm_lmseq = apply_normparameters(landmark_seq, *params)

    params = calculate_normparameters(all_seq)
    norm_all = apply_normparameters(all_seq, *params)

    # Saving datasets:
    sns.heatmap(norm_lmseq)
    plt.show()
    if not os.path.exists('files/norm/'):
        os.mkdir('files/norm/')
    np.save('files/norm/norm_lmseq.npy', norm_lmseq)
    np.save('files/norm/norm_allseq.npy', norm_all)


if __name__ == '__main__':
    main()
