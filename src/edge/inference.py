# load the edge model once and predict an intent for one utterance.
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_edge_model(checkpoint_dir, tokenizer_name, device="cpu"):
    # Function that loads the model and the tokenizer in the edge device
    # TODO: Tokenizer is loaded from the cloud, external dependency important to change it so it is local
    # TODO: Save the tokenizer along the checkpoint when re-training in the server, and load it from the directory here.
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()  # inference mode: disables dropout, deterministic outputs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer


def predict(model, tokenizer, labels, utterance, return_logits=False):
    # labels: {int index -> intent name}, matching the model's current head
    # return_logits: also return the raw logits, needed by logit-based OOD scores (Energy)
    device = next(model.parameters()).device
    inputs = tokenizer(
        utterance,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",  # important returning a pytorch tensor
    ).to(device)

    with torch.no_grad():  # no gradient tracking to save memory and time
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze(
        0
    )  # turning the logits into a probability per intent
    pred_idx = int(
        torch.argmax(probs).item()
    )  # picks the index the index with the highes probability

    name = labels[pred_idx]  # get the intent name of the prediction
    confidence = float(probs[pred_idx].item())
    if return_logits:
        return (
            name,
            confidence,
            probs.tolist(),
            logits.squeeze(0).tolist(),
        )  # return the logits as well
    return (
        name,
        confidence,
        probs.tolist(),
    )  # ood needs to track the whole probability distribution rather than the single predicted intent


def get_logits(model, tokenizer, utterances, batch_size=64):
    """
    Same forward pass as predict(), over many utterances at once: used to score a whole split,
    by the server when it calibrates the OOD threshold and by the detection benchmark.
    Returns an array of shape [n utterances, num_labels].
    """
    device = next(model.parameters()).device
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(utterances), batch_size):
            batch = utterances[start : start + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            ).to(device)
            all_logits.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(all_logits)
