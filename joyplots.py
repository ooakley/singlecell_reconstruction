"""
Friday afternoon and I wanted to do something fun.
Joyplots for the avgTomoseq data!

python joyplots.py 
"""

import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def main():
    # Importing tomoSeq data and converting to numpy array:
    tomo_array = pd.read_csv('files/raw/embryo_average_landmark_genes_landmarkEbar.csv')
    tomo_array = tomo_array.loc[:, '1':]
    tomo_array = tomo_array.to_numpy(dtype='float32', copy=True)
    print(tomo_array.shape)
    tomo_array = list(tomo_array)
    x_range = list(range(100))
    x_ticks = list(range(0,101,20))
    _, axes = joypy.joyplot(tomo_array, kind='values', x_range=x_range, fade=True, overlap=0.5)
    axes[-1].set_xticks(x_ticks)
    plt.show()

if __name__ == '__main__':
    main()
