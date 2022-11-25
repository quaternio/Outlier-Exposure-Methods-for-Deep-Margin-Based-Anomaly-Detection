# Outlier Exposure Methods for Deep Margin-Based Anomaly Detection

Outlier exposure (OE) has been shown to be an effective method to improve anomaly detection performance at test time [1]. The method presented in [1] uses logit suppression via KL-divergence between the model’s softmax distribution and the uniform distribution. A potential alternative to this method is to aggregate all out-of-distribution instances into a single “outlier class” during training time.

Both of these methods are compatible with a variety of loss functions. Among these, the margin loss [2]  is of interest. We propose a set of experiments considering these outlier exposure methods with cross-entropy and margin losses.

The following is the experiment matrix that we're interested in:

|             | **Cross-Entropy** | **Margin Loss** |
|:------------|:------------------|:----------------|
|**Logit Suppression**| | |
|**Kitchen Sink**| | |

This draws heavily from the paper linked below.

## [[arxiv]](https://arxiv.org/abs/1803.05598) [[Official TF Repo]](https://github.com/google-research/google-research/tree/master/large_margin)

[1]  D. Hendrycks, M. Mazeika, and T. Dietterich, Deep Anomaly Detection with Outlier Exposure. arXiv, 2018. doi: 10.48550/ARXIV.1812.04606.

[2] G. F. Elsayed, D. Krishnan, H. Mobahi, K. Regan, and S. Bengio, Large Margin Deep Networks for Classification. arXiv, 2018. doi: 10.48550/ARXIV.1803.05598.

<hr>

## Results 

### Coming Soon!
