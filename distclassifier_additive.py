"""
Script to define and train a distributional classifier with pytorch.

Example:
python distclassifier_additive.py
"""
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import spearmanr
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
        self.relu = nn.ReLU(inplace=False)
        self.lin1 = nn.Linear(numgene, 150)
        self.lin2 = nn.Linear(150, 100)
        self.sm = nn.Softmax(1)
        self.lsm = nn.LogSoftmax(1)
        mean = tomo.mean(1, keepdim=True)
        std = tomo.std(1, keepdim=True)
        self.norm_tomo = (tomo-mean)/std
        # tomo_max = scaled_tomo.max(1, keepdim=True)[0]
        # tomo_min = scaled_tomo.min(1, keepdim=True)[0]
        # = (scaled_tomo - tomo_min)/(tomo_max - tomo_min)

    def split_cell(self, norm):
        net_out = self.lin2(self.dropout(self.relu(self.lin1(norm))))
        classifier_output = self.sm(torch.unsqueeze(net_out, 0))
        return classifier_output

    def forward(self, norm, raw):
        split = self.split_cell(norm)
        with torch.no_grad():
            self.prediction_history.append(split.detach().numpy())
        split_dist = torch.mm(torch.unsqueeze(raw, 1), split)
        self.generated_map = self.generated_map + split_dist
        self.class_entropy = -torch.sum(split * split.log())

    def calculate_loss(self, loss_func):
        gene_mean = (self.generated_map + 1e-30).mean(1, keepdim=True)
        gene_std = (self.generated_map + 1e-30).std(1, keepdim=True)
        self.norm_map = (self.generated_map-gene_mean)/gene_std
        print(torch.isnan(self.generated_map).any())
        # self.norm_map[torch.isnan(self.norm_map)] = 0
        bin_mean = self.sm(self.generated_map.mean(0, keepdim=True))
        bin_entropy = -torch.sum(bin_mean * bin_mean.log())
        bin_entropy[torch.isnan(bin_entropy)] = 0
        self.loss = loss_func(
            self.norm_map, self.norm_tomo
            ) + 0*self.class_entropy - 30*bin_entropy

    def backward(self):
        self.loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= p.grad * self.learning_rate
            self.zero_grad()

    def initialise_distribution(self, norm, raw):
        self.generated_map = torch.zeros(48, 100)
        self.prediction_history = []


def calculate_correlation(predictions, rawseq, tomoseq):
    assignment = predictions.argmax(axis=1)
    assignment = np.expand_dims(assignment, 1)
    assembled = np.zeros((48, 100))
    for i in range(100):
        cell_indices = np.nonzero(assignment == i)
        binned_cells = rawseq.numpy()[cell_indices, :]
        if binned_cells.shape[1] == 0:
            continue
        assembled[:, i] = np.mean(binned_cells)
    corr_array = []
    for i in range(48):
        correlation, _ = spearmanr(assembled[i, :], tomoseq.numpy()[i, :])
        corr_array.append(correlation)
    mean_corr = np.mean(np.asarray(corr_array))
    return mean_corr


def fit(model, loss_func, loss_list, corr_list, j, norm, raw, tomo):
    for i in range(50):
        # Generating map:
        model.initialise_distribution(norm, raw)

        # Backprop:
        model.forward(norm[i, :], raw[i, :])
        model.calculate_loss(loss_func)
        if torch.isnan(model.loss):
            model.zero_grad()
            continue
        model.backward()
        loss_list.append(model.loss.item())

        #  Readout:
        if i % 3000 == 0:
            print('>>>> Epoch: ' + str(j+1))
            print('>>>> Loss: ' + str(model.loss.item()))
            plt.ion()
            plt.close()
            fig, axs = plt.subplots(2)
            axs[0].plot(np.stack(loss_list))
            if len(corr_list) != 0:
                axs[1].plot(np.stack(corr_list))
            plt.show()
            plt.pause(0.000001)

    with torch.no_grad():
        model.prediction_history = np.concatenate(
            model.prediction_history, 0)
        corr_list.append(
            calculate_correlation(model.prediction_history, raw, tomo))


def main():
    """
    Import data, fit model, visualise model,
    save trained model weights.
    """
    EPOCHS = 1
    timestamp = datetime.today()
    timestamp = str(timestamp)[0:16]
    torch.manual_seed(0)
    np.random.seed(0)

    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/norm/smooth_lmtomo.npy')[0:48, :]  # (48, 100)
    normseq = np.load('files/norm/norm_varseq.npy')[:, 0:50]
    rawseq = np.load('files/norm/norm_lmseq.npy')[:, 0:48]

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    norm_tensor = torch.from_numpy(normseq)
    raw_tensor = torch.from_numpy(rawseq)

    # Instantiating model:
    dist_classifier = DistClassifier(
        1e-10, 0.5, tomo_tensor, normseq.shape[1])
    loss_func = nn.MSELoss(reduction='mean')

    # Fitting model:
    print('Fitting model...')
    loss_list = []
    corr_list = []
    for j in range(EPOCHS):
        tensors = (norm_tensor, raw_tensor, tomo_tensor)
        fit(dist_classifier, loss_func, loss_list, corr_list, j, *tensors)
        rng_state = np.random.get_state()
        np.random.shuffle(norm_tensor)
        np.random.set_state(rng_state)
        np.random.shuffle(raw_tensor)

    # Generating directories:
    if not os.path.exists('files/training_plots/'):
        os.mkdir('files/training_plots/')
    if not os.path.exists('files/final_maps/'):
        os.mkdir('files/final_maps/')

    # Saving loss history:
    plt.ioff()
    plt.close()
    plt.clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(np.stack(loss_list))
    axs[1].plot(np.stack(corr_list))
    plt.savefig('files/training_plots/' + timestamp)
    plt.close()

    # Saving final generated map:
    sns.heatmap(dist_classifier.generated_map.detach().numpy())
    plt.savefig('files/final_maps/' + timestamp + ' additive')
    plt.close()
    sns.heatmap(dist_classifier.norm_tomo)
    plt.savefig('files/final_maps/tomo')

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
