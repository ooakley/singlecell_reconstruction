"""
Script to generate figures for my presentation.

Example:
python presentation_figures.py
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.signal import savgol_filter
import os

if not os.path.exists('files/presentation_images/'):
    os.mkdir('files/presentation_images/')

# Loading data:
variable_tomo = np.load('files/variable_tomo.npy')
RCC = np.load('files/RCClandmarkvalues.npy')
sc_dataset = np.load('files/48seqcounts.npy')
var_seq = np.load('files/variable_dataset.npy')
normvar_seq = np.load('files/norm/norm_varseq.npy')
smoothlmtomo = np.load('files/norm/smooth_lmtomo.npy')
lmtomo = np.load('files/avgtomodata.npy')
fiftytomo = np.load('files/fiftybintomoseq.npy')

# Plotting variable tomo:
fig, ax = plt.subplots()
ax = sns.heatmap(variable_tomo, cbar=False)
ax.set(xlabel='section number (A->P)', ylabel='gene',
       title='Tomoseq data')
fig.savefig("files/presentation_images/variablegene_tomoseq.png", dpi=300)
fig.tight_layout()
plt.close()

# Plotting RCC landmark values:
RCC_assignment = RCC.argmax(axis=1)
RCC_assembled = np.zeros((48, 50))
for i in range(50):
    cell_indices = np.nonzero(RCC_assignment == i)
    binned_cells = sc_dataset[cell_indices, :]
    if binned_cells.shape[1] == 0:
        continue
    RCC_assembled[:, i] = np.mean(binned_cells, axis=1)

mean = np.expand_dims(np.mean(RCC_assembled, axis=1), 1)
std = np.expand_dims(np.std(RCC_assembled, axis=1), 1)
RCC_assembled = (RCC_assembled-mean)/std
RCC_assembled = savgol_filter(RCC_assembled, window_length=5, polyorder=2)

RCCcorr_array = []
for i in range(48):
    correlation, _ = spearmanr(RCC_assembled[i, :], fiftytomo[i, :])
    RCCcorr_array.append(correlation)
RCCscore = np.median(np.asarray(RCCcorr_array))
print('RCC score: ' + str(RCCscore))

fig, ax = plt.subplots()
ax = sns.heatmap(RCC_assembled, cbar=False)
ax.set(xlabel='section number (A->P)', ylabel='gene',
       title='RCC reconstruction  - landmark genes')
fig.tight_layout()
fig.savefig("files/presentation_images/RCC reconstruction.png", dpi=300)
plt.close()

# Plotting some representative RCC traces:
fig, ax = plt.subplots()
ax.plot(RCC[3, :])
ax.set(xlabel='section number (A->P)', ylabel='RCC value',
       title='RCC profile  - single cell')
fig.tight_layout()
fig.savefig("files/presentation_images/RCCprofile2.png", dpi=300)
plt.close()

# Plotting effects of normalisation:
fig, ax = plt.subplots()
ax = sns.heatmap(var_seq, cbar=True)
ax.set(xlabel='genes', ylabel='individual cells',
       title='Variable genes from single cell dataset')
fig.tight_layout()
fig.savefig("files/presentation_images/varseq.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
ax = sns.heatmap(normvar_seq, cbar=True)
ax.set(xlabel='genes', ylabel='individual cells',
       title='Normalised variable genes from single cell dataset')
fig.tight_layout()
fig.savefig("files/presentation_images/normvarseq.png", dpi=300)
plt.close()

fig, ax = plt.subplots()
ax = sns.heatmap(lmtomo, cbar=True)
ax.set(xlabel='section number (A->P)', ylabel='gene',
       title='Landmark genes from tomoseq dataset')
fig.tight_layout()
fig.savefig("files/presentation_images/smoothtomo.png", dpi=300)
plt.close()

mean = np.expand_dims(np.mean(lmtomo, axis=1), 1)
std = np.expand_dims(np.std(lmtomo, axis=1), 1)
norm_lmtomo = (lmtomo-mean)/std

fig, ax = plt.subplots()
ax = sns.heatmap(norm_lmtomo, cbar=True)
ax.set(xlabel='section number (A->P)', ylabel='gene',
       title='Normalised landmark genes from tomoseq dataset')
fig.tight_layout()
fig.savefig("files/presentation_images/normsmoothtomo.png", dpi=300)
plt.close()
