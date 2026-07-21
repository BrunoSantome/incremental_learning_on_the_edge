from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


def build_model(name, n_labels=60):
    """
    Simple method to build the model and return the associated tokenizer

    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=n_labels
    )
    return model, tokenizer, model.num_parameters(only_trainable=True)


def compute_class_weights(dataloader, num_labels, scheme="sqrt_inv", device="cpu"):
    """Inverse-frequency class weights, normalised to mean 1.0."""
    counts = np.zeros(num_labels, dtype=np.int64)
    for batch in dataloader:
        labels = batch["labels"].numpy()
        counts += np.bincount(labels, minlength=num_labels)

    counts = np.maximum(counts, 1)  # guard empty classes
    if scheme == "sqrt_inv":
        w = 1.0 / np.sqrt(counts)  # gentler, recommended default
    elif scheme == "none":
        w = np.ones(num_labels)
    else:
        raise ValueError(scheme)

    w = w / w.mean()  # keeps loss magnitude comparable to plain CE
    return torch.tensor(w, dtype=torch.float, device=device)
