# Experimental Results

The following tables summarize the outcomes captured in `results.csv`.

## Bank Marketing

| method | hyperparameters | accuracy | precision | recall | f1 | duration |
|--------|-----------------|---------:|----------:|-------:|----:|---------:|
| Decision Tree | {'random_state': 42} | 0.874 | 0.877 | 0.874 | 0.875 | 0.286 |
| MLPClassifier | {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42} | 0.892 | 0.887 | 0.892 | 0.889 | 29.526 |

## Books Reviews

| method | hyperparameters | accuracy | precision | recall | f1 | duration |
|--------|-----------------|---------:|----------:|-------:|----:|---------:|
| Logistic Regression | {'max_iter': 1000} | 0.875 | 0.875 | 0.875 | 0.875 | 0.038 |

## TF Flowers

| method | hyperparameters | accuracy | precision | recall | f1 | duration |
|--------|-----------------|---------:|----------:|-------:|----:|---------:|
| Simple CNN | {'epochs': 3} | 0.586 | 0.600 | 0.586 | 0.568 | 49.937 |

## Discussion

For the **Bank Marketing** dataset, the classical Decision Tree baseline (best configuration from `baseline_dt.py` uses `criterion='entropy', max_depth=5`) achieved around 0.895 accuracy. The MLP neural network from `main.py` slightly improved accuracy to about 0.892 but required roughly 30 seconds to train compared with under a second for the Decision Tree. While the neural model offers a modest performance gain, it comes at a much higher runtime cost.

On the **Books Reviews** text dataset, Logistic Regression provided the best results (0.875 accuracy) and finished in well under a second. Decision Tree baselines peaked near 0.72 accuracy, so classical linear methods remain preferable here. No neural network was evaluated for this dataset.

For **TF Flowers** images, classical Decision Trees on color histograms reached at most 0.425 accuracy, whereas the simple CNN trained for three epochs achieved 0.586 accuracy. Training the CNN took close to one minute on CPU, illustrating the significant runtime trade-off for higher accuracy on image data.

Across datasets the pattern is clear: classical models train almost instantly, making them attractive when computation time is limited. Neural networks yield better performance on complex data such as images but require considerably longer runtimes, even for modest architectures.
