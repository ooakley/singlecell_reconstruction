# singlecell_reconstruction

 Reconstructing spatial distribution of single cell RNA seq data using machine learning.

 So far, two attempts have been made to improve upon RCC-based assignment of cells to
 different A/P sections:

 1. Forcing a neural network to learn a lossy encoding of something like an
 RCC, then seeing whether this encoding allows for better assignment of cells.
 Sadly this didn't really work.

 2. An entirely RCC independent model, in which a kind of nested bootstrapping
 occurs (what I've named a 'distributional classifier'). In order to make the assignment
 of a cell to an A/P section differentiable, instead of putting all of that cell's counts into
 a single bin, the cell's counts are proportionally distributed across the 50 bin
 A/P axis according to the softmax activations in the final layer. This is used to generate
 an initial map. Individual cells are then passed through the classifier and the
 loss is calculated by adding their distribution to the previously generated distribution
 and comparing it to the normalised tomoseq data (different comparison functions are
 currently undergoing testing.)
