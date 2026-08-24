import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .distillation_1 import _get_incremental_version_dir
import pandas as pd


def evaluate_test_set(model, test_dataloader, device=None):
    """
    Method that mimics the evaluation method during training, used to evaluate a model on the test split.

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model, test_dataloader, device=None, id2intent=None, num_labels=None
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
        average=None,  # no macro, no weighted,
        zero_division=0,
    )

    per_intent_f1 = {}

    for label_id in all_label_ids:
        key = id2intent[label_id] if id2intent else label_id
        per_intent_f1[key] = float(f1_per_label[label_id])

    return metrics, per_intent_f1


def intents_report(dataclass, student_key, config, n_versions, device=None):
    """
    Load each version's checkpoint (_v0.._v{n}) and score it on the TEST split
    restricted to the intents it knows.

    This has to be perform after training, and the dataclass must be the same object altered by training
    so the incremental intents have been added.
    """

    tokenizer = AutoTokenizer.from_pretrained(config[student_key]["name"])
    id2intent = dataclass.id2intent
    test_split = dataclass.sets_names[1]  # "test_set"

    rows = {}  # version -> {intent_name: f1}
    metrics_by_version = {}  # version -> aggregate metrics (macro/weighted f1, acc, loss, per-language)
    for n in range(n_versions + 1):  # V0 .. Vn
        checkpoint = _get_incremental_version_dir(config, student_key, n)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        num_labels = model.config.num_labels  # 15 + n
        loader = dataclass.build_split_loader(
            test_split, tokenizer, student_key, max_label=num_labels
        )
        metrics, per_intent = evaluate_per_intent(
            model, loader, device, id2intent, num_labels
        )
        rows[n] = per_intent
        metrics_by_version[n] = metrics
        print(f"V{n}: has {num_labels} intents on the test split")
    return rows, metrics_by_version


def old_intent_persistance_table(rows, id2intent, n_original=15):
    """
    Table of the original intents F1 across versions + a mean-old row (the retention
    forgetting number)
    """

    original = [id2intent[i] for i in range(n_original)]
    data = {f"V{n}": {name: rows[n].get(name) for name in original} for n in rows}
    df = pd.DataFrame(data)
    df.loc["mean_old"] = df.mean()
    return df


def new_intent_acquisition_table(rows, id2intent, n_original=15):
    """
    Table of the new additional intents F1 across versions
    old_intent_persistance_table (old) vs new_intent_acquisition_table (new)
    """
    n_versions = max(rows.keys())
    new_names = [id2intent[i] for i in range(n_original, n_original + n_versions)]
    data = {f"V{n}": {name: rows[n].get(name) for name in new_names} for n in rows}
    df = pd.DataFrame(data)
    df.loc["mean_new"] = df.mean()
    return df
