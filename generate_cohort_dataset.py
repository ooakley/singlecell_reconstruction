"""
Generate subselections of RNAseq data,
convert to numpy and save locally to files.


Example:
python generate_cohort_dataset.py 50 True
"""
import os
import random
import argparse
import numpy as np


def main():
    # Parsing arguments:
    parser = argparse.ArgumentParser(
        description='Script to generate population subselections dataset.')
    parser.add_argument('size', type=int, help='Size of resultant dataset.')
    parser.add_argument('all', type=bool, help='Whether to include all genes.')
    args = parser.parse_args()

    # Seeding for reproducibility:
    np.random.seed(seed=10)
    random.seed(10)

    # Importing data:
    if all:
        sc_array = np.load('files/norm/norm_all.npy')
    else:
        sc_array = np.load('files/norm/norm_subset.npy')

    # Generating subselections:
    dataset = []
    for dummy_iterator in range(args.size):
        selection_indices = random.sample(range(6189), 1000)
        selections = sc_array[np.array(selection_indices), :]
        dataset.append(selections)
    dataset = np.stack(dataset, axis=0)
    print(dataset.shape)

    # Saving to file:
    if not os.path.exists('files/'):
        os.mkdir('files/')
    if all:
        np.save(
            'files/norm/' + str(args.size) + 'alldatasubselections.npy',
            dataset
            )
    else:
        np.save(
            'files/norm/' + str(args.size) + 'landmarkdatasubselections.npy',
            dataset
            )


if __name__ == '__main__':
    main()
