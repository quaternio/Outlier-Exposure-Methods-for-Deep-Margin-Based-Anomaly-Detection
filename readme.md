# Thomas Noel's Master's Degree Project

Outlier exposure (OE) has been shown to be an effective method to improve anomaly detection performance at test time [1]. The method presented in [1] uses logit suppression via KL-divergence between the model’s softmax distribution and the uniform distribution. A potential alternative to this method is to aggregate all out-of-distribution instances into a single “outlier class” during training time.

Both of these methods are compatible with a variety of loss functions. Among these, the margin loss [2]  is of interest. We propose a set of experiments considering these outlier exposure methods with cross-entropy and margin losses.

The following is the experiment matrix that we're interested in:

|             | **Cross-Entropy** | **Margin Loss** |
|:------------|:------------------|:----------------|
|**Logit Suppression**| | |
|**Kitchen Sink**| | |

This draws heavily from the paper linked below.

## [[arxiv]](https://arxiv.org/abs/1803.05598) [[Official TF Repo]](https://github.com/google-research/google-research/tree/master/large_margin)

<hr>

## Results 

### Coming Soon!