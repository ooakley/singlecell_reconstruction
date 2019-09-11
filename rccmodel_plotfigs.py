"""
A script to plot individual cell profiles from
a trained RCC generator model.

Example:
python rccmodel_plotfigs.py
"""
# pylint: disable=import-error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def main():
    # Importing data and model:
    sc_dataset = np.load('files/normalised48counts.npy')
    RCC_dataset = np.load('files/RCClandmarkvalues.npy')
    tomoseq_dataset = np.load('files/fiftybintomoseq.npy')
    model = load_model('files/models/2019-09-10 12:39.h5')

    # Generating RCC predictions:
    predictions_dataset = model.predict(sc_dataset)

    # Plotting individual cell actual + predicted RCC trace:
    fig, ax = plt.subplots()
    ax.plot(predictions_dataset[48, :], label='Predicted')
    ax.plot(RCC_dataset[48, :], label='Actual')
    ax.set(xlabel='A/P sections', ylabel='Arbitrary correlation units',
        title='Comparing RCC and model output')
    ax.grid()
    ax.legend()

    fig.savefig('files/images/individualtrace.png')
    plt.show()

if __name__ == '__main__':
    main()
