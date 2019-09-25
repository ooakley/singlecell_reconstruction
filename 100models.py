"""
100 models in three days. I guess we'll see how it goes lol.

Example:
python 100models.py
"""
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os


class section_classifier(nn.Module):
    def __init__(self, section, dropout_rate, learning_rate, numgene):
        super().__init__()
        # Internal values:
        self.section = torch.unsqueeze(section, 1)
        self.learning_rate = learning_rate

        # Internal tensors and functions:
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(inplace=False)
        self.lin1 = nn.Linear(numgene, 10)
        self.lin2 = nn.Linear(10, 1)
        self.sm = nn.Softmax(1)
        self.loss_func = nn.MSELoss(reduction='mean')

    def init_state_variables(self):
        # Internal variables of state:
        self.aggregate_selections = torch.zeros(48, 1)
        self.proportion_counter = torch.zeros(1)

    def forward(self, input, raw):
        self.value = self.lin2(self.dropout(self.relu(self.lin1(input))))
        dist = torch.unsqueeze(raw, 1) * self.value
        alt = self.aggregate_selections + dist
        self.norm_alt = alt/(self.proportion_counter + 1e-20)
        with torch.no_grad():
            # plt.close()
            # fig, axs = plt.subplots(2)
            # axs[0].plot(self.aggregate_selections/self.value)
            # axs[1].plot(self.section)
            # plt.show()
            # plt.pause(0.00000001)
            self.aggregate_selections += dist
            self.proportion_counter += self.value

    def calculate_loss(self):
        self.loss = self.loss_func(self.norm_alt, self.section)

    def backward(self):
        torch.autograd.set_detect_anomaly(True)
        self.loss.backward()
        with torch.no_grad():
            for p in self.parameters():
                p -= p.grad * self.learning_rate
            self.zero_grad()


def fit(model_list, epochs, norm, raw):
    loss_list = []
    for i in range(epochs):
        print('>>>> Epoch: ' + str(i+1))
        loss_instance = []
        for j in range(len(model_list)):
            if (j+1) % 25 == 0:
                print('Section: ' + str(j+1))
            model_list[j].init_state_variables()
            for k in range(20):
                model_list[j].forward(norm[k, :], raw[k, :])
                model_list[j].zero_grad()
            for k in range(norm.shape[0]-20):
                model_list[j].forward(norm[k+20, :], raw[k+20, :])
                model_list[j].calculate_loss()
                model_list[j].backward()
            loss_instance.append(model_list[j].loss.item())
        loss_list.append(np.mean(np.asarray(loss_instance)))
    return loss_list


def main():
    """Train model."""
    epochs = 10
    timestamp = datetime.today()
    timestamp = str(timestamp)[0:16]
    torch.manual_seed(0)
    np.random.seed(0)

    # Importing data:
    print('Loading data...')
    lmtomo = np.load('files/norm/smooth_lmtomo.npy')[0:48, 0:100]  # (48, 100)
    normseq = np.load('files/norm/norm_varseq.npy')[0:100, 0:100]
    rawseq = np.load('files/48seqcounts.npy')[0:2000, 0:48]

    # Converting to torch tensors:
    print('Converting numpy arrays to torch tensors...')
    tomo_tensor = torch.from_numpy(lmtomo)
    norm_tensor = torch.from_numpy(normseq)
    raw_tensor = torch.from_numpy(rawseq)

    # Instantiate model collection:
    print('Instantiating collection of models...')
    model_list = []
    dropout_rate = 0.1
    learning_rate = 0.001

    numgene = normseq.shape[1]
    for i in range(lmtomo.shape[1]):
        section = tomo_tensor[:, i]
        model_list.append(
            section_classifier(section, dropout_rate, learning_rate, numgene))

    # Train:
    print('Training model...')
    plt.ion()
    loss_list = fit(model_list, epochs, norm_tensor, raw_tensor)
    plt.ioff()

    # Outputting results:
    plt.plot(np.asarray(loss_list))
    plt.savefig('files/training_plots/100models ' + timestamp)
    plt.close()
    reconstructed_tomoseq = []
    for i in range(len(model_list)):
        section = model_list[i].aggregate_selections
        proportion = model_list[i].proportion_counter
        reconstructed_tomoseq.append((section/proportion).numpy())
    reconstructed_tomoseq = np.concatenate(reconstructed_tomoseq, 1)
    sns.heatmap(reconstructed_tomoseq)
    plt.savefig('files/final_maps/' + timestamp + ' 100model')


if __name__ == '__main__':
    main()
