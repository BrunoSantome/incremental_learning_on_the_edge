# decide whether a prediction is unknown. Unknown detection mechanism
import numpy as np


# Simplest ood possible, to test the pipeline ood. A high value of threshold will escalte almost every time, just
# to test the whole end-to-end pìpeline. This must be improved with a more complex ood mechanism TODOç
# Softmax Confidence Thresholding
def is_ood(probs, threshold=0.95):
    return max(probs) < threshold


# OOD scores computed from the raw logits
# They accept a single utterance or a batch for the benchmark.


def _logsumexp(x, axis=-1):
    # shift by the max so exp() does not overflow
    # following paper: Accurately computing the log-sum-exp and softmax functions
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


# Baseline, similar to is_ood but using logits to compute
def msp_score(logits):
    # Maximum Softmax Probability (Hendrycks & Gimpel, 2017)
    x = np.asarray(logits, dtype=np.float64)
    return np.exp(
        np.max(x, axis=-1) - _logsumexp(x)
    )  # max softmax = exp(max logit - logsumexp)
