"""
Script to define and train a distributional classifier with pytorch.

Example:
python distributional_classifier.py
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns


class DistClassifier(nn.Module):
    """
    Class to encapsulate methods and architecture of
    distributional classifier neural net.
    """
    def __init__(self, learning_rate, dropout_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lin1 = nn.Linear(48, 50)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.lin2 = nn.Linear(50, 50)
        self.sm = nn.Softmax(0)
        self.lsm = nn.LogSoftmax(0)

    def split_cell(self, input):
        classifier_output = self.sm(
            self.lin2(self.dropout1(self.lin1(input))))
        return classifier_output

    def forward(self, all_input, landmark_input):
        split_dist = torch.mm(
            torch.unsqueeze(landmark_input, 1),
            torch.unsqueeze(self.split_cell(all_input), 0)
            )
        self.alt_map = self.generated_map + split_dist

    def calculate_loss(self, loss_func, tomo_data):
        self.loss = loss_func(self.alt_map, tomo_data)

    def backward(self):
        self.loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= p.grad * self.learning_rate
            self.zero_grad()

    def generate_distribution(self, all, landmark):
        self.generated_map = torch.zeros(48, 50)
        with torch.no_grad():
            for i in range(all.shape[0]):
                split_dist = torch.mm(
                    torch.unsqueeze(landmark[i, :], 1),
                    torch.unsqueeze(self.split_cell(all[i, :]), 0)
                    )
                self.generated_map += split_dist


def main():
    """
    Import data, fit model, visualise model,
    save trained model weights.
    """

    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/norm/axisnorm_lmtomo.npy')
    lmseq = np.load('files/norm/axisnorm_lmseq.npy')
    # all_array = np.load('files/norm/norm_all.npy')

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    landmark_tensor = torch.from_numpy(lmseq)
    print(landmark_tensor.shape[0])
    # all_tensor = torch.from_numpy(all_array)[:, 0:500]

    # Training model:
    dist_classifier = DistClassifier(0.001, 0.1)
    loss_func = nn.MSELoss(reduction='sum')

    # Fitting model:
    print('Fitting model...')
    loss_list = []
    for i in range(landmark_tensor.shape[0]):
        if i % 500 == 0:
            dist_classifier.generate_distribution(
                landmark_tensor, landmark_tensor)
        dist_classifier.forward(landmark_tensor[i, :],
                                landmark_tensor[i, :])
        dist_classifier.calculate_loss(loss_func, tomo_tensor)
        dist_classifier.backward()
        loss_list.append(dist_classifier.loss.item())

        if i+1 % 100 == 0:
            print('>>>> Epoch: ' + str(i+1))
            print('>>>> Loss: ' + str(dist_classifier.loss.item()))
            plt.ion()
            plt.clf()
            plt.show()
            plt.plot(np.stack(loss_list))
            plt.pause(0.000001)

    # Saving model output:
    plt.ioff()
    error_map = np.square(
        np.subtract(dist_classifier.generated_map, lmtomo)
        )
    sns.heatmap(error_map)
    plt.show()
    model_predictions = []
    with torch.no_grad():
        for i in range(landmark_tensor.shape[0]):
            model_predictions.append(
                dist_classifier.split_cell(landmark_tensor[i, :]).numpy()
                )
    model_predictions = np.stack(model_predictions, 0)
    print(model_predictions.shape)
    if not os.path.exists('files/predictions/'):
        os.mkdir('files/predictions/')
    np.save('files/predictions/distclassifier_predictions.npy',
            model_predictions)


if __name__ == '__main__':
    main()
