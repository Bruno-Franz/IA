# Experimental Results

This document summarizes the metrics obtained when executing `main.py` and `baseline_dt.py` on the three datasets included in the project.

## Bank Marketing (tabular)

Classical method tested was a Decision Tree with three configurations. The best configuration in terms of accuracy was **Config 2** (`criterion='entropy', max_depth=5`) achieving **0.895** accuracy. Config 1 yielded slightly lower accuracy but better recall and F1 for the minority class.

The neural approach used a simple **Decision Tree** from `main.py` (no neural network was trained for this dataset). Training took less than a second.

## Books Reviews (text)

For text classification a **Logistic Regression** model achieved **0.875** accuracy with `max_iter=1000`. Runtime was under a second.

No neural network architecture was implemented for this dataset.

## TF Flowers (images)

The classical baseline uses Decision Trees on color histograms; **Config 2** again gave the highest accuracy of **0.425**. A simple convolutional neural network was trained for three epochs, reaching **0.616** accuracy and took about three minutes on CPU.

## Observations

- Classical models train almost instantly (under one second) while the CNN requires several minutes, highlighting the runtime trade-off for moderate gains in accuracy.
- Preprocessing the Books Reviews data required ensuring the CSV path matches the repository layout. The Flowers dataset depends on `tensorflow-datasets`; downloading and processing images was slow without GPU support.

