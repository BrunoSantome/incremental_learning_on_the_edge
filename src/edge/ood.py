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


def energy_score(logits, T=1.0):
    # Energy (Liu et al., 2020): negative free energy T * logsumexp(logits / T)
    # T > 1 smooths the logits, it can help the distilled student (trained with soft targets)
    # following paper Classical Out-of-Distribution Detection Methods Benchmark in Text Classification Tasks
    x = np.asarray(logits, dtype=np.float64)
    return T * _logsumexp(x / T)


def calibrate_threshold(scores, keep=0.95):
    # threshold from known-intent scores only (eval split), set to 95 by default
    # example: 95% of known utterances pass and 5% get escalated.
    # this is thought to include the energy score since it isnt between 1 and 0.
    return float(np.percentile(np.asarray(scores, dtype=np.float64), (1 - keep) * 100))


def is_ood_score(score, threshold):
    # same as is_ood but for any score
    return score < threshold
