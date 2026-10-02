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
from core.evaluate import (
    evaluate_per_intent,
    intents_report,
    old_intent_persistance_table,
    new_intent_acquisition_table,
)
from experiments.config import (
    ExperimentConfig,
    checkpoint_resolver,
    is_done,
    save_result,
    save_run_config,
)
from experiments.synthetic_cache import (
    cache_path,
    ensure_synthetic_utterances,
    escalation_cache_path,
    ensure_escalated_intents,
)

CONDITIONS = {
    # name: (use_kd, use_replay, select_best)
    # naive keeps the model of the last epoch: its eval split covers the old intents, which this
    # condition is defined not to have access to, so selecting an epoch with it would itself
    # mitigate the forgetting the condition exists to expose (and it did: selection kept an early
    # epoch at mean_old 0.93 while the final model had collapsed to 0.03).
    # Rule: a condition's data access governs both its training and its checkpoint selection.
    # naive and kd store nothing of the old intents, so they also get no old-intent eval split to
    # select an epoch with (validation rows are stored old examples too) and keep the last epoch.
    # replay and full do hold a buffer, so they use the system's own selection criterion.
    "naive": (False, False, False),
    "replay": (False, True, True),
    "kd": (True, False, False),
    "full": (True, True, True),
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


def _config_for(
    condition,
    seed,
    config,
    K=70,
    student_key="student1",
    data_source="real",
    n_intents=5,
):
    """The ExperimentConfig of one condition: same run_id for training and for reporting."""
    use_kd, use_replay, _ = CONDITIONS[condition]
    distill_cfg = config[student_key]["distill"]
    return ExperimentConfig(
        experiment="ablation",
        condition=condition,
        use_kd=use_kd,
        use_replay=use_replay,
        seed=seed,
        K=K,
        alpha=distill_cfg["alpha"] if use_kd else 0.0,
        w=distill_cfg["selection_new_weight"],
        data_source=data_source,
        n_intents=n_intents,
        student_key=student_key,
    )


def run_condition(
    condition,
    seed,
    config,
    K=70,
    n_versions=None,
    student_key="student1",
    data_source="real",
    escalations_file=None,
):
    """
    One chain V1..Vn for a single condition and seed.

    data_source="synthetic" trains the new intents on the cached LLM-generated utterances
    (eval/test stay real, so the numbers remain comparable with the real-data runs). The cache is
    shared by every condition, so the generated data is identical across them.

    escalations_file (data_source="escalated") runs the long-horizon scenario: a JSON array of
    escalated utterances, each one turned into an intent the LLM names itself, with train, eval and
    test all generated. Retention is still measured on the 15 original intents' real test data.
    """
    use_kd, use_replay, select_best = CONDITIONS[condition]
    distill_cfg = config[student_key]["distill"]

    _reset_registry(config)
    dataclass = DataClass()  # rebuilt at the 15 pretrain intents

    synthetic = None  # {intent: [utterances]}, real eval/test
    escalated = None  # {intent: [utterances]}, eval/test generated too
    if escalations_file:
        data_source = "escalated"
        escalated = ensure_escalated_intents(
            dataclass, config, escalations_file, seed=seed
        )
        intents = list(escalated)
    else:
        intents = reserve_intent_names(dataclass, config)
    if n_versions is not None:
        intents = intents[:n_versions]
    if data_source == "synthetic":
        synthetic = ensure_synthetic_utterances(dataclass, config, intents, seed=seed)

    cfg = _config_for(
        condition, seed, config, K, student_key, data_source, n_intents=len(intents)
    )
    resolve = checkpoint_resolver(cfg, config)

    save_run_config(
        cfg,
        extra={
            "intents_in_order": intents,
            "student": config[student_key]["name"],
            "temperature": distill_cfg["temperature"],
            "select_best": select_best,  # False -> final-epoch checkpoint (naive baseline)
            "synthetic_cache": cache_path(seed) if synthetic else None,
            "escalation_cache": (
                escalation_cache_path(escalations_file, seed) if escalated else None
            ),
        },
    )
    print(f"\n=== {cfg.run_id} ===\nintents: {intents}")

    for version, intent_name in enumerate(intents, start=1):
        # synthetic: LLM train rows, real eval/test | escalated: train+eval+test all generated
        if synthetic:
            new_utt = dataclass.build_llm_new_utt(intent_name, synthetic[intent_name])
        elif escalated:
            new_utt = dataclass.build_synthetic_splits(
                intent_name, escalated[intent_name], n_eval=20, n_test=20, seed=seed
            )
        else:
            new_utt = None  # real reserve data

        if is_done(cfg, version):
            # already trained in an earlier session: keep the label space in sync and move on
            dataclass.admit_intent(intent_name, new_utt=new_utt)
            print(f"v{version} ({intent_name}): already done, skipped")
            continue

        print(
            f"v{version}: adding '{intent_name}'"
            f" [kd={use_kd}, replay={use_replay}, data={data_source}]"
        )
        started = time.perf_counter()
        output_dir = run_incremental_step(
            dataclass,
            intent_name,
            student_key,
            config,
            version,
            K,
            seed,
            new_utt=new_utt,  # None -> real reserve data
            wandb_group=cfg.run_id,
            wandb_tags=[condition, f"K{K}", f"seed{seed}", data_source, intent_name],
            use_kd=use_kd,
            use_replay=use_replay,
            select_best=select_best,
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
                "data_source": data_source,
                "num_labels": num_labels,
                "train_seconds": seconds,
                "per_intent_f1": per_intent,
                "metrics": metrics,
            },
        )
        print(f"v{version}: {seconds:.0f}s, saved {path}")

    return cfg


def report_condition(
    condition,
    seed,
    config,
    K=70,
    student_key="student1",
    data_source="real",
    escalations_file=None,
):
    _reset_registry(config)
    dataclass = DataClass()

    if escalations_file:  # long-horizon scenario: names and data come from the cache
        data_source = "escalated"
        escalated = ensure_escalated_intents(
            dataclass, config, escalations_file, seed=seed
        )
        intents = list(escalated)
        for intent_name in intents:
            dataclass.admit_intent(
                intent_name,
                new_utt=dataclass.build_synthetic_splits(
                    intent_name, escalated[intent_name], n_eval=20, n_test=20, seed=seed
                ),
            )
    else:
        intents = reserve_intent_names(dataclass, config)
        for intent_name in intents:
            dataclass.admit_intent(intent_name)

    cfg = _config_for(
        condition, seed, config, K, student_key, data_source, n_intents=len(intents)
    )
    resolve = checkpoint_resolver(cfg, config)

    rows, metrics_by_version = intents_report(
        dataclass,
        student_key,
        config,
        n_versions=len(intents),
        checkpoint_dir=resolve,  # this condition's own checkpoints, shared V0 at version 0
    )
    retention = old_intent_persistance_table(rows, dataclass.id2intent)
    acquisition = new_intent_acquisition_table(rows, dataclass.id2intent)
    print(f"\n=== {cfg.run_id} ===")
    # to_string(): prints every version column, pandas would otherwise drop the middle ones
    print(f"\nretention (old intents)\n{retention.to_string()}")
    print(f"\nacquisition (new intents)\n{acquisition.to_string()}")
    return rows, metrics_by_version


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
    parser.add_argument(
        "--data",
        default="real",
        choices=["real", "synthetic"],
        help="new-intent training data: real MASSIVE rows, or the cached LLM-generated "
        "utterances (eval/test stay real either way)",
    )
    parser.add_argument(
        "--escalations",
        default=None,
        help="JSON array of escalated utterances (long-horizon scenario): the LLM names and "
        "generates each intent, train/eval/test all synthetic. Overrides --data",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="do not train: re-score the checkpoints already trained and print the "
        "retention / acquisition tables of each condition",
    )
    args = parser.parse_args()

    config = load_config()
    for seed in args.seeds:
        for condition in args.conditions:
            if args.report:
                report_condition(
                    condition,
                    seed,
                    config,
                    K=args.K,
                    data_source=args.data,
                    escalations_file=args.escalations,
                )
            else:
                run_condition(
                    condition,
                    seed,
                    config,
                    K=args.K,
                    n_versions=args.versions,
                    data_source=args.data,
                    escalations_file=args.escalations,
                )


if __name__ == "__main__":
    main()
