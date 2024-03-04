# Classification of NN architectures for temporal data
The Neural Net architectures designed for Temporal Data have two ways of categorization:

1. In terms of what they can predict:
   - `sequence-to-point` predictors: these take a sequence of timepoints and output a point-predictor (which could contain more than one label).
   - `sequence-to-sequence` predictors: these output a predictor label for each timepoint (possibly excluding some warm-up segment).
2. In terms of how the data are passed through them:
   - `recurrent` architecture: data points are sequentially fed through a network and a label is output for every time point.
   - `non-recurrent` architecture: data points are collected into segments and passed jointly through a network.
