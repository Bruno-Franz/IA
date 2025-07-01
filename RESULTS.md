# Experimental Results

The following tables summarize the outcomes captured in `results.csv`.

## Bank Marketing

| method | hyperparameters | accuracy | precision | recall | f1 | duration |
|--------|-----------------|---------:|----------:|-------:|----:|---------:|
| Decision Tree | {'random_state': 42} | 0.874 | 0.877 | 0.874 | 0.875 | 0.379 |
| MLPClassifier | {'hidden_layer_sizes': (100,), 'max_iter': 300, 'random_state': 42} | 0.896 | 0.889 | 0.896 | 0.892 | 15.358 |

## Books Reviews

| method | hyperparameters | accuracy | precision | recall | f1 | duration |
|--------|-----------------|---------:|----------:|-------:|----:|---------:|
| Logistic Regression | {'max_iter': 1000} | 0.875 | 0.875 | 0.875 | 0.875 | 0.016 |

## Discussion

On the **Bank Marketing** dataset, the MLP neural network outperformed the classical
Decision Tree, achieving **0.896 accuracy** versus **0.874**. The trade‑off was
a much longer runtime: around **15&nbsp;seconds** for the MLP compared with
under **0.4&nbsp;seconds** for the Decision Tree.

For the **Books Reviews** text data we only evaluated a classical model.
The Logistic Regression classifier reached **0.875 accuracy** in roughly
**0.016&nbsp;seconds** and no neural model results were recorded for comparison.

Overall, the neural approach gave the highest score on the tabular task but
required significantly more time to train, whereas the classical method for
sentiment analysis ran nearly instantly and already produced strong results.
