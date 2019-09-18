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
    def __init__(self, learning_rate, dropout_rate, tomo, numgene):
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.lin1 = nn.Linear(numgene, (numgene+50)//2)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.lin2 = nn.Linear((numgene+50)//2, 50)
        self.sm = nn.Softmax(1)
        mean = tomo.mean(1, keepdim=True)
        std = tomo.std(1, keepdim=True)
        self.scaled_tomo_map = (tomo-mean)/std

    def split_cell(self, norm):
        net_out = self.lin2(self.dropout1(self.lin1(norm)))
        classifier_output = self.sm(torch.unsqueeze(net_out, 0))
        return classifier_output

    def forward(self, norm, raw):
        split_dist = torch.mm(
            torch.unsqueeze(raw, 1),
            self.split_cell(norm)
            )
        self.alt_map = self.generated_map + split_dist

    def calculate_loss(self, loss_func):
        mean = self.alt_map.mean(1, keepdim=True)
        std = self.alt_map.std(1, keepdim=True)
        norm_map = (self.alt_map-mean)/std
        self.loss = loss_func(norm_map, self.scaled_tomo_map)

    def backward(self):
        self.loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= p.grad * self.learning_rate
            self.zero_grad()

    def generate_distribution(self, norm, raw):
        self.generated_map = torch.zeros(48, 50)
        with torch.no_grad():
            for i in range(norm.shape[0]):
                split_dist = torch.mm(
                    torch.unsqueeze(raw[i, :], 1),
                    self.split_cell(norm[i, :])
                    )
                self.generated_map += split_dist


def fit(norm, raw, model, loss_func, loss_list):
    for i in range(norm.shape[0]):
        # Generating map:
        if i % 500 == 0:
            model.generate_distribution(norm, raw)

        # Backprop:
        model.forward(norm[i, :], raw[i, :])
        model.calculate_loss(loss_func)
        model.backward()
        loss_list.append(model.loss.item())

        #  Readout:
        if i % 500 == 0:
            print('>>>> Epoch: ' + str(i))
            print('>>>> Loss: ' + str(model.loss.item()))
            plt.ion()
            plt.clf()
            plt.show()
            plt.plot(np.stack(loss_list))
            plt.pause(0.000001)


def main():
    """
    Import data, fit model, visualise model,
    save trained model weights.
    """
    EPOCHS = 2
    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/fiftybintomoseq.npy')  # (48, 50)
    normseq = np.load('files/allseqcounts.npy')[:, 0:1000]  # (6189, 23946)
    rawseq = np.load('files/48seqcounts.npy')

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    norm_tensor = torch.from_numpy(normseq)
    raw_tensor = torch.from_numpy(rawseq)

    # Training model:
    dist_classifier = DistClassifier(
        0.1, 0.1, tomo_tensor, normseq.shape[1])
    loss_func = nn.MSELoss(reduction='mean')

    # Fitting model:
    print('Fitting model...')
    loss_list = []
    for j in range(EPOCHS):
        fit(norm_tensor, raw_tensor, dist_classifier, loss_func, loss_list)

    # Saving model output:
    model_predictions = []
    with torch.no_grad():
        for i in range(norm_tensor.shape[0]):
            model_predictions.append(
                dist_classifier.split_cell(norm_tensor[i, :]).numpy()
                )
    model_predictions = np.concatenate(model_predictions, 0)
    print(model_predictions.shape)
    if not os.path.exists('files/predictions/'):
        os.mkdir('files/predictions/')
    np.save('files/predictions/distclassifier_predictions.npy',
            model_predictions)


if __name__ == '__main__':
    main()
