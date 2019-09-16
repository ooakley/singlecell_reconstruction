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
    # Importing data:
    landmark_seq = np.load('files/48seqcounts.npy')
    all_seq = np.load('files/allseqcounts.npy')
    avgtomodata = np.load('files/fiftybintomoseq.npy')

    # Normalising data:
    norm_seq = normalise(landmark_seq)
    print('Landmark genes: ' + str(norm_seq.shape))
    norm_all = normalise(all_seq)
    print('All genes: ' + str(norm_all.shape))
    norm_tomo = normalise(avgtomodata)
    print('Tomo array: ' + str(norm_tomo.shape))

    # Saving datasets:
    if not os.path.exists('files/norm/'):
        os.mkdir('files/norm/')
    np.save('files/norm/norm_subset.npy', norm_seq)
    np.save('files/norm/norm_all.npy', norm_all)
    np.save('files/norm/norm_tomo.npy', norm_tomo)


if __name__ == '__main__':
    main()
