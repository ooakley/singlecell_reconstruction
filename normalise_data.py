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
    mean = np.mean(dataset, axis=1)
    mean = np.expand_dims(mean, axis=1)
    print(mean.shape)
    std = np.std(dataset, axis=1)
    std = np.expand_dims(std, axis=1)
    print(std.shape)
    normalised_dataset = (dataset - mean)/std
    max = np.max(normalised_dataset, axis=1)
    max = np.expand_dims(max, axis=1)
    min = np.min(normalised_dataset, axis=1)
    min = np.expand_dims(min, axis=1)
    return mean, std, max, min


def main():
    # Importing data:
    landmark_tomodata = np.load('files/fiftybintomoseq.npy')
    print('lmtomo shape: ' + str(landmark_tomodata.shape))
    landmark_seq = np.load('files/48seqcounts.npy')
    print('lmseq shape: ' + str(landmark_seq.shape))

    # TODO: all gene normalisation
    # all_tomodata = np.load('files/allseqcounts.npy')
    # all_seq = np.load('files/allseqcounts.npy')

    # Normalising landmark data:
    params = calculate_normparameters(landmark_tomodata)
    norm_lmtomo = apply_normparameters(landmark_tomodata, *params)
    print('Tomo array: ' + str(norm_lmtomo.shape))
    norm_lmseq = apply_normparameters(np.transpose(landmark_seq), *params)
    norm_lmseq = np.transpose(norm_lmseq)
    print('Landmark genes: ' + str(norm_lmseq.shape))

    # norm_all = apply_normparameters(all_seq, params)
    # print('All genes: ' + str(norm_all.shape))

    # Saving datasets:
    sns.heatmap(norm_lmtomo)
    plt.show()
    if not os.path.exists('files/norm/'):
        os.mkdir('files/norm/')
    np.save('files/norm/axisnorm_lmseq.npy', norm_lmseq)
    np.save('files/norm/axisnorm_lmtomo.npy', norm_lmtomo)
    # np.save('files/norm/norm_all.npy', norm_all)


if __name__ == '__main__':
    main()
