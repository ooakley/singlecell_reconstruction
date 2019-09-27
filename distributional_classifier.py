"""
Script to define and train a distributional classifier with pytorch.

Example:
python distributional_classifier.py
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
        self.relu = nn.LeakyReLU(inplace=False)
        self.lin1 = nn.Linear(numgene, 500)
        self.lin2 = nn.Linear(500, 100)
        self.sm = nn.Softmax(1)
        self.lsm = nn.LogSoftmax(1)
        mean = tomo.mean(1, keepdim=True)
        std = tomo.std(1, keepdim=True)
        self.norm_tomo = self.sm((tomo-mean)/std)
        # tomo_max = self.norm_tomo.max(1, keepdim=True)[0]
        # tomo_min = self. norm_tomo.min(1, keepdim=True)[0]
        # self.norm_tomo = (self.norm_tomo - tomo_min)/(tomo_max - tomo_min)

    def split_cell(self, norm):
        lin1out = self.dropout(self.relu(self.lin1(norm)))
        lin2out = self.lin2(lin1out)
        net_out = lin2out
        classifier_output = self.sm(torch.unsqueeze(net_out, 0))
        return classifier_output

    def forward(self, norm, raw):
        split = self.split_cell(norm)
        split_dist = torch.mm(torch.unsqueeze(raw, 1), split)
        self.alt_map = self.generated_map + split_dist
        self.alt_predmap = self.predictions_map + split
        self.class_entropy = -torch.sum(split * (split+1e-20).log())
        print(self.class_entropy.item())

    def calculate_loss(self, loss_func):
        self.alt_map = self.alt_map / self.alt_predmap
        gene_mean = self.alt_map.mean(1, keepdim=True)
        gene_std = (self.alt_map+1e-20).std(1, keepdim=True)
        self.norm_map = self.sm((self.alt_map-gene_mean)/gene_std)
        sm_bins = self.sm(self.alt_predmap)
        bin_entropy = -torch.sum(sm_bins * (sm_bins+1e-20).log())
        # map_max = self.norm_map.max(1, keepdim=True)[0]
        # map_min = self.norm_map.min(1, keepdim=True)[0]
        # self.norm_map = (self.norm_map - map_min)/(map_max - map_min)
        self.loss = loss_func(
            self.norm_map, self.norm_tomo
            ) + 0.1*(((2-self.class_entropy)**2)) - 0.5*bin_entropy

    def backward(self):
        self.loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= p.grad * self.learning_rate
            self.zero_grad()

    def generate_distribution(self, norm, raw):
        self.generated_map = torch.zeros(48, 100)
        self.predictions_map = torch.zeros(1, 100)
        self.prediction_history = []
        with torch.no_grad():
            for i in range(200):
                split_cell = self.split_cell(norm[i, :])
                split_dist = torch.mm(
                    torch.unsqueeze(raw[i, :], 1), split_cell
                    )
                self.prediction_history.append(split_cell.numpy())
                self.generated_map += split_dist
                self.predictions_map += split_cell
            self.prediction_history = np.concatenate(
                self.prediction_history, 0)


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
    for i in range(5):
        # Generating map:
        if i % 500 == 0:
            model.generate_distribution(norm, raw)
            with torch.no_grad():
                corr_list.append(
                    calculate_correlation(model.prediction_history, raw, tomo))

        # Backprop:
        model.forward(norm[i, :], raw[i, :])
        model.calculate_loss(loss_func)
        model.backward()
        loss_list.append(model.loss.item())

        #  Readout:
        if i % 500 == 0:
            print('>>>> Epoch: ' + str(j+1))
            print('>>>> Loss: ' + str(model.loss.item()))
            plt.ion()
            plt.close()
            fig, axs = plt.subplots(2)
            axs[0].plot(np.stack(loss_list))
            axs[1].plot(np.stack(corr_list))
            plt.show()
            plt.pause(0.000001)


def main():
    """
    Import data, fit model, visualise model,
    save trained model weights.
    """
    EPOCHS = 50
    timestamp = datetime.today()
    timestamp = str(timestamp)[0:16]
    torch.manual_seed(0)
    np.random.seed(0)

    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/norm/smooth_lmtomo.npy')[0:48, :]  # (48, 100)
    normseq = np.load('files/norm/norm_varseq.npy')[:, 0:600]
    rawseq = np.load('files/48seqcounts.npy')[:, 0:48]

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    norm_tensor = torch.from_numpy(normseq)
    raw_tensor = torch.from_numpy(rawseq)

    # Instantiating model:
    dist_classifier = DistClassifier(
        0.1, 0.5, tomo_tensor, normseq.shape[1])
    loss_func = nn.MSELoss(reduction='mean')

    # Fitting model:
    print('Fitting model...')
    loss_list = []
    corr_list = []
    with torch.autograd.detect_anomaly():
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
    # plt.ioff()
    plt.close()
    plt.clf()
    fig, axs = plt.subplots(2)
    axs[0].plot(np.stack(loss_list))
    axs[1].plot(np.stack(corr_list))
    plt.savefig('files/training_plots/' + timestamp)
    plt.close()

    # Saving final generated map:
    sns.heatmap(dist_classifier.norm_map.detach().numpy())
    plt.savefig('files/final_maps/' + timestamp + 'norm')
    plt.close()
    sns.heatmap(dist_classifier.alt_map.detach().numpy())
    plt.savefig('files/final_maps/' + timestamp + 'raw')
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
    print(dist_classifier.predictions_map)


if __name__ == '__main__':
    main()
