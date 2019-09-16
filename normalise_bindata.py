"""
A script to normalise all necessary data for the training of a neural network.

Example:
python normalise_bindata.py
"""
import os
import numpy as np


def normalise(dataset):
    normalised_dataset = (dataset - np.mean(dataset))/np.std(dataset)
    normalised_dataset = (normalised_dataset - np.min(normalised_dataset))/(
        np.max(normalised_dataset) - np.min(normalised_dataset))
    return normalised_dataset


def main():
    # Importing data.
    landmark_seq = np.load('files/48seqcounts.npy')
    all_seq = np.load('files/allseqcounts.npy')
    avgtomodata = np.load('files/fiftybintomoseq.npy')

    # Normalising data:
    norm_seq = normalise(landmark_seq)
    norm_all = normalise(all_seq)
    norm_tomo = normalise(avgtomodata)

    # Saving datasets:
    if not os.path.exists('files/norm/'):
        os.mkdir('files/norm/')
    np.save('files/norm/norm_subset.npy', norm_seq)
    np.save('files/norm/norm_all.npy', norm_all)
    np.save('files/norm/norm_tomo.npy', norm_tomo)
