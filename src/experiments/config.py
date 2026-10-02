# Identity of one experimental run: what varies between runs, and where its files go.
# Every run writes under its own run_id, so two conditions (or two seeds) can never overwrite
# each other's checkpoints or results.

import json
import os
from dataclasses import dataclass, asdict

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(SRC_DIR, "outputs", "checkpoints")
RESULTS_DIR = os.path.join(SRC_DIR, "outputs", "results")


@dataclass
class ExperimentConfig:
    """
    One run of one experiment. condition names which mechanism is active, and use_kd / use_replay
    are what the incremental step actually reads:
      naive  -> no replay buffer, no distillation (the fine-tuning baseline)
      replay -> replay buffer only (alpha = 0)
      kd     -> distillation only
      full   -> both (the adopted method)
      joint  -> all intents trained together, upper bound
    """

    experiment: str  # "ablation" | "synthetic" | "e2e"
    condition: str = "full"
    use_kd: bool = True
    use_replay: bool = True
    seed: int = 42
    K: int = 70  # replay buffer: utterances kept per known intent
    alpha: float = 0.2  # KD weight in the loss
    w: float = 0.3  # selection_new_weight, checkpoint selection criterion
    # "real" (MASSIVE reserve intents) | "synthetic" (LLM utterances, real eval/test)
    # | "escalated" (LLM names and generates the intent from one escalated utterance, all splits synthetic)
    data_source: str = "real"
    n_intents: int = 5  # increments this run covers (the horizon)
    student_key: str = "student1"
    note: str = ""  # free text, ends up in run_config.json

    @property
    def run_id(self):
        # the horizon is only tagged when it differs from the 5 reserve intents, so the runs
        # finished before this field existed keep their directories
        horizon = "" if self.n_intents == 5 else f"_n{self.n_intents}"
        return (
            f"{self.experiment}_{self.condition}_{self.data_source}"
            f"_K{self.K}_a{self.alpha}_w{self.w}{horizon}_s{self.seed}"
        )


def v0_checkpoint(config, student_key="student1"):
    """
    The deployed V0 model: shared by every condition, so it stays outside the run directories
    and is never retrained by an experiment.
    """
    return os.path.join(SRC_DIR, config[student_key]["distill"]["output_dir"]) + "_v0"


def checkpoint_dir(cfg, version):
    """Where version Vn of this run is saved (V0 is shared, see v0_checkpoint)."""
    return os.path.join(CHECKPOINTS_DIR, cfg.run_id, f"v{version}")


def checkpoint_resolver(cfg, config):
    """
    version -> checkpoint path, for this run: the shared V0 for version 0, this run's own
    directory afterwards. Passed to intents_report(checkpoint_dir=...) and used as
    previous_dir / output_dir in run_incremental_step.
    """

    def resolve(version):
        if version == 0:
            return v0_checkpoint(config, cfg.student_key)
        return checkpoint_dir(cfg, version)

    return resolve


def run_dir(cfg):
    return os.path.join(RESULTS_DIR, cfg.run_id)


def result_path(cfg, version):
    """One JSON per version, written as soon as the version is evaluated."""
    return os.path.join(run_dir(cfg), f"v{version}.json")


def is_done(cfg, version):
    """True if this version already ran: lets a runner resume after a Colab disconnection."""
    return os.path.exists(result_path(cfg, version))


def save_result(cfg, version, payload):
    os.makedirs(run_dir(cfg), exist_ok=True)
    with open(result_path(cfg, version), "w") as f:
        json.dump(payload, f, indent=2)
    return result_path(cfg, version)


def load_result(cfg, version):
    with open(result_path(cfg, version)) as f:
        return json.load(f)


def save_run_config(cfg, extra=None):
    """
    Snapshot of what produced this run (the dataclass + anything else worth pinning, e.g. the
    resolved model names), so results stay interpretable months later.
    """
    os.makedirs(run_dir(cfg), exist_ok=True)
    payload = asdict(cfg)
    if extra:
        payload.update(extra)
    path = os.path.join(run_dir(cfg), "run_config.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
