import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .distillation_1 import _get_incremental_version_dir


def evaluate_test_set(model, test_dataloader, device):
    """
    Method that mimics the evaluation method during training, used to evaluate a model on the test split.

    """
    model.eval()
    model.to(device)

    losses = []
    y_true = []
    y_predicted = []
    y_locales = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="evaluating test set"):
            locales = batch.pop("locale")
            if "utt" in batch:
                batch.pop("utt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device.type):
                outputs = model(**batch)
            losses.append(outputs.loss.item())
            y_predicted += outputs.logits.argmax(-1).cpu().tolist()
            y_true += batch["labels"].cpu().tolist()
            y_locales += list(locales)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_predicted)
    loc_arr = np.array(y_locales)

    metrics = {
        "test_loss": sum(losses) / len(losses),
        "test_accuracy": accuracy_score(y_true, y_predicted),
        "test_macro_f1": f1_score(y_true, y_predicted, average="macro"),
        "test_weighted_f1": f1_score(y_true, y_predicted, average="weighted"),
    }
    for language in np.unique(loc_arr):
        mask = loc_arr == language
        metrics[f"test_accuracy_{language}"] = accuracy_score(
            y_true_arr[mask], y_pred_arr[mask]
        )
        metrics[f"test_macro_f1_{language}"] = f1_score(
            y_true_arr[mask], y_pred_arr[mask], average="macro"
        )
        metrics[f"test_weighted_f1_{language}"] = f1_score(
            y_true_arr[mask], y_pred_arr[mask], average="weighted"
        )
    return metrics, y_true, y_predicted


def evaluate_per_intent(
    model, test_dataloader, device, id2intent=None, num_labels=None
):
    """
    Per-intent F1. Reuses evaluate_test_set, adds one F1 per class in index order.
    """
    metrics, true_labels, predicted_labels = evaluate_test_set(
        model, test_dataloader, device
    )

    if num_labels is None:
        num_labels = model.config.num_labels

    all_label_ids = list(range(num_labels))
    f1_per_label = f1_score(
        true_labels,
        predicted_labels,
        labels=all_label_ids,
        average=None,
        zero_division=0,
    )

    per_intent_f1 = {}
    for label_id in all_label_ids:
        key = id2intent[label_id] if id2intent else label_id
        per_intent_f1[key] = float(f1_per_label[label_id])

    return metrics, per_intent_f1
