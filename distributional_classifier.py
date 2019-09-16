"""
Script to define and train a distributional classifier with pytorch.

Example:
python distributional_classifier.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistClassifier(nn.Module):
    """
    Class to encapsulate methods and architecture of
    distributional classifier neural net.
    """
    def __init__(self):
        super().__init__()
        self.dropout_rate = 0.1
        self.lin1 = nn.Linear(48, 50)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.lin2 = nn.Linear(50, 50)

    def split_cell(self, input):
        classifier_output = nn.Softmax(
            self.lin2(self.dropout1(self.lin1(input))))
        split_dist = torch.mm(input, classifier_output)
        return split_dist

    def forward(self, input):
        split_dist = self.split_cell(input)
        alt_map = self.generated_map + split_dist
        return alt_map

    def generate_distribution(self, subselections):
        self.generated_map = torch.zeros(48, 50)
        with torch.no_grad():
            for i in range(subselections.shape[0]):
                self.generated_map += self.split_cell(subselections[i, :])


def main():
    """
    Import data, fit model, visualise model,
    save trained model weights.
    """
    # Importing data:
    subselection_array = np.load('files/norm/1datasubselections.npy')
    tomo_data = np.load('files/norm/norm_tomo.npy')

    # Training model:
    dist_classifier = DistClassifier()
    loss_func = F.KLDivLoss

    # Evaluating model:

    # Saving model weights:


if __name__ == '__main__':
    main()
