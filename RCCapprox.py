"""
Generate and train model to approximate RCC distributions from single cell data.

Example:
python RCCapprox.py
"""
#pylint: disable=import-error
import os
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.compat.v1.keras import backend as K

def define_model():

    model = Sequential()
    rate = 0.1

    model.add(Dense(100, input_dim=48, activation='relu'))
    model.add(Dropout(rate, noise_shape=None, seed=None))
    model.add(BatchNormalization())

    model.add(Dense(50))
    model.add(Dropout(rate, noise_shape=None, seed=None))
    model.add(BatchNormalization())

    model.add(Dense(10))
    model.add(Dropout(rate, noise_shape=None, seed=None))
    model.add(BatchNormalization())

    model.add(Dense(50))

    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mse')

    return model

def main():
    """
    Seed, import, etc.
    """
    # Generating timestamp:
    timestamp = datetime.today()
    timestamp = str(timestamp)[0:16]

    # Seed model:
    seed_value = 10
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.compat.v1.random.set_random_seed(seed_value)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    K.set_session(sess)

    # Import data:
    sc_dataset = np.load('files/normalised48counts.npy')
    RCC_dataset = np.load('files/RCClandmarkvalues.npy')
    tomoseq_dataset = np.load('files/fiftybintomoseq.npy')

    # Define model:
    model = define_model()
    model.fit(sc_dataset, RCC_dataset, epochs=3000, batch_size=100, verbose=2, validation_split=0.1)

    # Plotting results:
    if not os.path.exists('files/images/'):
        os.mkdir('files/images/')
    model_predictions = model.predict(sc_dataset)
    _ = sns.heatmap(model_predictions[0:100, :])
    plt.savefig('files/images/predictions_heatmap.png', dpi=300, orientation='portrait')
    plt.show()
    plt.clf()
    _ = sns.heatmap(RCC_dataset[0:100, :])
    plt.savefig('files/images/RCC_heatmap.png', dpi=300, orientation='portrait')
    plt.show()
    plt.clf()

    # Saving model:
    if not os.path.exists('files/models/'):
        os.mkdir('files/models/')
    model.save('files/models/' + timestamp + '.h5')


if __name__ == '__main__':
    main()
