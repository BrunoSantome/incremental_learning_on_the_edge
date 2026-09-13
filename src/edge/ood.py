# decide whether a prediction is unknown. Unknown detection mechanism

# Simplest ood possible, to test the pipeline ood. A high value of threshold will escalte almost every time, just
# to test the whole end-to-end pìpeline. This must be improved with a more complex ood mechanism TODO
def is_ood(probs, threshold=0.8):
    return max(probs) < threshold 