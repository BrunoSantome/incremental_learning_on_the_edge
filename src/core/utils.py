from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import torch.nn as nn


def build_model(name, n_labels):
    """
    Simple method to build the model and return the associated tokenizer.

    ``n_labels`` must be passed explicitly (e.g. ``DataClass.num_labels``) so the
    classifier head matches the intents present in the current phase instead of a
    fixed, oversized label space.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=n_labels,
        torch_dtype=torch.float32,
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


def expand_student_head(model, n_new):
    """
    Method used to grow the head of a classifier from n_old to n_old + n_new,
    preserving the weights of the n_old classes and randomely initializing the n_new one.

    Warm start
    """
    n_old = model.config.num_labels
    hidden = model.classifier.in_features
    n_total = n_old + n_new

    old_classifier = model.classifier
    new_classifier = nn.Linear(hidden, n_total)  # init randomly all rows
    with (
        torch.no_grad()
    ):  # this copies from the old_classifier weights and bias into the new classifier
        new_classifier.weight[:n_old] = old_classifier.weight
        new_classifier.bias[:n_old] = old_classifier.bias

    model.classifier = new_classifier  # this is specific to the student model
    model.config.num_labels = n_total  # we need to update also this, since we access the variable in setup_label_space
    model.num_labels = n_total
    return model
