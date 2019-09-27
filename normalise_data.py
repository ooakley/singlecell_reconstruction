"""
A script to normalise all necessary data for the training of a neural network.

Example:
python normalise_data.py
"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def apply_normparameters(dataset, mean, std, max, min):
    dataset = np.clip(dataset, 0, 80)
    normalised_dataset = (dataset - mean)/std
    normalised_dataset = (normalised_dataset - min)/(max - min)
    return normalised_dataset


def calculate_normparameters(dataset):
    dataset = np.clip(dataset, 0, 80)
    mean = np.mean(dataset, axis=0)
    print(mean.shape)
    mean = np.expand_dims(mean, axis=0)
    std = np.std(dataset, axis=0) + 1e-20
    std = np.expand_dims(std, axis=0)
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
    print(landmark_seq.max())

    all_seq = np.load('files/allseqcounts.npy')
    print('all_seq shape: ' + str(all_seq.shape))
    print(all_seq.max())

    var_seq = np.load('files/variable_dataset.npy')
    print('var_seq shape: ' + str(var_seq.shape))
    print(var_seq.max())

    lm_tomo = np.load('files/avgtomodata.npy')
    print('lmtomo shape: ' + str(lm_tomo.shape))

    fifty_tomo = np.load('files/fiftybintomoseq.npy')
    print('fiftytomo shape: ' + str(fifty_tomo.shape))

    # Normalising landmark data:
    params = calculate_normparameters(landmark_seq)
    norm_lm = apply_normparameters(landmark_seq, *params)

    params = calculate_normparameters(all_seq)
    norm_all = apply_normparameters(all_seq, *params)

    params = calculate_normparameters(var_seq)
    norm_var = apply_normparameters(var_seq, *params)

    # Smoothing tomoseq data:
    smooth_tomo = savgol_filter(lm_tomo, window_length=11, polyorder=2)
    smooth_fiftytomo = savgol_filter(fifty_tomo, window_length=5, polyorder=2)

    # Saving datasets:
    sns.heatmap(norm_var)
    plt.show()
    sns.heatmap(norm_lm)
    plt.show()
    if not os.path.exists('files/norm/'):
        os.mkdir('files/norm/')
    np.save('files/norm/norm_lmseq.npy', norm_lm)
    np.save('files/norm/norm_allseq.npy', norm_all)
    np.save('files/norm/norm_varseq.npy', norm_var)
    np.save('files/norm/smooth_lmtomo', smooth_tomo)
    np.save('files/norm/smooth_fiftytomo', smooth_fiftytomo)


if __name__ == '__main__':
    main()
