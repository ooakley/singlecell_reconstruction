"""
Generate subselections of RNAseq data,
convert to numpy and save locally to files.

Generate 50 section bins of avgTomoSeq data.

Example:
python generate_cohort_dataset.py 20
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
    args = parser.parse_args()

    # Seeding for reproducibility:
    np.random.seed(seed=10)
    random.seed(10)

    # Importing data:
    sc_array = np.load('files/normalised48counts.npy')

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
    np.save('files/' + str(args.size) + 'datasubselections.npy', dataset)


if __name__ == '__main__':
    main()
