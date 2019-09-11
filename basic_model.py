"""
Define and train a basic model in pytorch, where a distribution is built from a set of cells. 
Mainly to help me learn pytorch and how to construct a loss function from the tomoseq dataset.

Example:
python basic_model.py
"""
import numpy as np
import torch

class basic_model:
    def __init__(self, targetdist, dataset):
        self.targetdist = targetdist
        self.dataset = dataset
        D_in, H, D_out = 1000, 100, 50

    def forward(self):
        return None
    def backward(self):
        return None
    def save_weights(self, filepath)
        return None
    def load_weights(self, filepath)
        return None

def main():
    # Importing data:
    sc_dataset = np.load('files/1datasubselections.npy')
    tomo_dataset = np.load('files/avgtomodata.npy')

    # Seeding 
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Neural network definition:

    # Saving logs and trained model:

    # 
    return None

if __name__ == '__main__':
    main()
