"""
Script to define and train a distributional classifier with pytorch.

Example:
python distributional_classifier.py
"""
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


class DistClassifier(nn.Module):
    """
    Class to encapsulate methods and architecture of
    distributional classifier neural net.
    """
    def __init__(self, learning_rate, dropout_rate, tomo, numgene):
        super().__init__()
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lin1 = nn.Linear(numgene, 2000)
        self.lin2 = nn.Linear(2000, 50)
        self.sm = nn.Softmax(1)
        self.lsm = nn.LogSoftmax(1)
        mean = tomo.mean(1, keepdim=True)
        std = tomo.std(1, keepdim=True)
        self.scaled_tomo_map = self.sm((tomo-mean)/std)

    def split_cell(self, norm):
        net_out = self.lin2(self.dropout(self.lin1(norm)))
        classifier_output = self.sm(torch.unsqueeze(net_out, 0))
        return classifier_output

    def forward(self, norm, raw):
        split = self.split_cell(norm)
        split_dist = torch.mm(torch.unsqueeze(raw, 1), split)
        self.alt_map = self.generated_map + split_dist
        self.entropy = -torch.sum(split * split.log())

    def calculate_loss(self, loss_func):
        mean = self.alt_map.mean(1, keepdim=True)
        std = self.alt_map.std(1, keepdim=True)
        norm_map = self.lsm((self.alt_map-mean)/std)
        self.loss = loss_func(
            norm_map, self.scaled_tomo_map) + 1e-6*self.entropy

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


def fit(norm, raw, model, loss_func, loss_list, j):
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
            print('>>>> Epoch: ' + str(j))
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
    EPOCHS = 20
    timestamp = datetime.today()
    timestamp = str(timestamp)[0:16]
    torch.manual_seed(0)
    np.random.seed(0)

    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/fiftybintomoseq.npy')  # (48, 50)
    normseq = np.load('files/norm/norm_varseq.npy')[:, 0:500]
    rawseq = np.load('files/48seqcounts.npy')

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    norm_tensor = torch.from_numpy(normseq)
    raw_tensor = torch.from_numpy(rawseq)

    # Instantiating model:
    dist_classifier = DistClassifier(
        0.1, 0.1, tomo_tensor, normseq.shape[1])
    loss_func = nn.KLDivLoss(reduction='batchmean')

    # Fitting model:
    print('Fitting model...')
    loss_list = []
    for j in range(EPOCHS):
        fit(norm_tensor, raw_tensor, dist_classifier, loss_func, loss_list, j)
        rng_state = np.random.get_state()
        np.random.shuffle(norm_tensor)
        np.random.set_state(rng_state)
        np.random.shuffle(raw_tensor)

    # Saving loss history:
    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(np.stack(loss_list))
    ax.set_title('Training loss')
    if not os.path.exists('files/training_plots/'):
        os.mkdir('files/training_plots/')
    plt.savefig('files/training_plots/' + timestamp)

    # Saving model predictions:
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
    np.save('files/predictions/dist_predictions_' + timestamp + '.npy',
            model_predictions)


if __name__ == '__main__':
    main()
