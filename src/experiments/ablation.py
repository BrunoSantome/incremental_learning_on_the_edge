# Forgetting ablation: how much of the retention comes from distillation, and how much from replay.
# Server-side training only, no detection / LLM / mailbox: the reserve MASSIVE intents are added in a
# fixed order with their real data, from the same shared V0 and the same seed, so the ONLY difference
# between conditions is the loss and the training set.
#
#   naive  -> no replay, no KD   (the standard fine-tuning baseline)
#   replay -> replay buffer only (alpha = 0)
#   kd     -> distillation only
#   full   -> both (the adopted method)
#
# Every version is evaluated and written to outputs/results/<run_id>/v{n}.json as soon as it is
# trained, so a Colab disconnection costs one version instead of the whole run: re-running skips
# whatever is already on disk.

import argparse
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.configuration import load_config
from core.dataloader import DataClass
from core.distillation_1 import run_incremental_step
from core.evaluate import evaluate_per_intent
from experiments.config import (
    ExperimentConfig,
    checkpoint_resolver,
    is_done,
    save_result,
    save_run_config,
)

CONDITIONS = {
    # name: (use_kd, use_replay, w)
    # w = checkpoint-selection weight; None keeps the configured value (0.3).
    # naive runs at w=1 (selection on the new intent only): picking the epoch by the old intents' eval scores would use information this condition is defined not to have, and would by itself
    # mitigate the forgetting it is meant to expose. The other conditions may use old-intent information by construction (replay buffer or teacher), so they keep the configured weight.
    "naive": (False, False, 1.0),
    "replay": (False, True, None),
    "kd": (True, False, None),
    "full": (True, True, None),
}


def reserve_intent_names(dataclass, config):
    """The intents to add, in the fixed order they appear in config.yaml (reproducible)."""
    return [dataclass.id2name[i] for i in config["dataset"]["reserve_intents"]]


def _reset_registry(config):
    """Back to the 15 pretrain intents, so every condition starts from V0's label space."""
    registry_path = DataClass._resolve_path(config["registry_path"])
    if os.path.exists(registry_path):
        os.remove(registry_path)


def evaluate_version(dataclass, config, student_key, checkpoint, device=None):
    """
    Score one version on the test split restricted to the intents it knows. Same computation as
    intents_report, for a single version, so each step pays for one evaluation only.
    """
    tokenizer = AutoTokenizer.from_pretrained(config[student_key]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    num_labels = model.config.num_labels
    loader = dataclass.build_split_loader(
        dataclass.sets_names[1],  # test_set
        tokenizer,
        student_key,
        max_label=num_labels,
    )
    metrics, per_intent = evaluate_per_intent(
        model, loader, device, dataclass.id2intent, num_labels
    )
    return metrics, per_intent, num_labels


def run_condition(
    condition, seed, config, K=70, n_versions=None, student_key="student1"
):
    """One chain V1..Vn for a single condition and seed."""
    use_kd, use_replay, w = CONDITIONS[condition]
    distill_cfg = config[student_key]["distill"]
    cfg = ExperimentConfig(
        experiment="ablation",
        condition=condition,
        use_kd=use_kd,
        use_replay=use_replay,
        seed=seed,
        K=K,
        alpha=distill_cfg["alpha"] if use_kd else 0.0,
        w=distill_cfg["selection_new_weight"] if w is None else w,
        student_key=student_key,
    )
    resolve = checkpoint_resolver(cfg, config)

    _reset_registry(config)
    dataclass = DataClass()  # rebuilt at the 15 pretrain intents
    intents = reserve_intent_names(dataclass, config)
    if n_versions is not None:
        intents = intents[:n_versions]

    save_run_config(
        cfg,
        extra={
            "intents_in_order": intents,
            "student": config[student_key]["name"],
            "temperature": distill_cfg["temperature"],
        },
    )
    print(f"\n=== {cfg.run_id} ===\nintents: {intents}")

    for version, intent_name in enumerate(intents, start=1):
        if is_done(cfg, version):
            # already trained in an earlier session: keep the label space in sync and move on
            dataclass.admit_intent(intent_name)
            print(f"v{version} ({intent_name}): already done, skipped")
            continue

        print(f"v{version}: adding '{intent_name}' [kd={use_kd}, replay={use_replay}]")
        started = time.perf_counter()
        output_dir = run_incremental_step(
            dataclass,
            intent_name,
            student_key,
            config,
            version,
            K,
            seed,
            wandb_group=cfg.run_id,
            wandb_tags=[condition, f"K{K}", f"seed{seed}", intent_name],
            use_kd=use_kd,
            use_replay=use_replay,
            w=w,
            previous_dir=resolve(version - 1),  # V0 is shared by every condition
            output_dir=resolve(version),
        )
        seconds = time.perf_counter() - started

        metrics, per_intent, num_labels = evaluate_version(
            dataclass, config, student_key, output_dir
        )
        path = save_result(
            cfg,
            version,
            {
                "run_id": cfg.run_id,
                "condition": condition,
                "seed": seed,
                "version": version,
                "intent_added": intent_name,
                "num_labels": num_labels,
                "train_seconds": seconds,
                "per_intent_f1": per_intent,
                "metrics": metrics,
            },
        )
        print(f"v{version}: {seconds:.0f}s, saved {path}")

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="forgetting ablation over the reserve intents"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["naive", "replay", "full"],
        choices=list(CONDITIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--K", type=int, default=70)
    parser.add_argument(
        "--versions",
        type=int,
        default=None,
        help="stop after this many intents (default: all reserve intents)",
    )
    args = parser.parse_args()

    config = load_config()
    for seed in args.seeds:
        for condition in args.conditions:
            run_condition(condition, seed, config, K=args.K, n_versions=args.versions)


if __name__ == "__main__":
    main()
