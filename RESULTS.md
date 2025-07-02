# Experimental Results

The following tables compile the metrics aggregated from `results.csv`.

## Bank Marketing

| method | accuracy | precision | recall | f1 | duration (s) |
|-------|---------:|----------:|-------:|---:|-------------:|
| Decision Tree | 0.874 | 0.877 | 0.874 | 0.875 | 0.379 |
| MLPClassifier | 0.896 | 0.889 | 0.896 | 0.892 | 15.358 |

## Books Reviews

| method | accuracy | precision | recall | f1 | duration (s) |
|-------------------|---------:|----------:|-------:|---:|-------------:|
| Logistic Regression | 0.875 | 0.875 | 0.875 | 0.875 | 0.016 |

## Discussion

**Bank Marketing.** Among classical approaches, the Decision Tree reached an accuracy of about 0.874 in under half a second. The neural MLPClassifier attained the best score, roughly 0.896 accuracy, but required more than 15&nbsp;seconds of training. This demonstrates a clear trade‑off between accuracy and runtime: the neural model is superior but far slower.

**Books Reviews.** Only the Logistic Regression baseline was recorded for this text dataset. It achieved around 0.875 accuracy in just 0.016&nbsp;seconds. No neural architecture results were available for comparison.

Overall, the experiments show that neural networks can deliver higher accuracy at the cost of significantly longer runtimes. When speed is crucial, the classical models may suffice, while neural methods become attractive if additional accuracy justifies the extra training time.
