from .utils import build_model, compute_class_weights, expand_student_head
from .configuration import load_config, set_seed
from .dataloader import DataClass
from transformers import (
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


"""
 Run_distillation function is to be changed but on notebook
 Confirm the teacher checkpoint's num_labels equals the student's registry count.

 Distillation is split into a shared BaseDistillationTrainer (training loop, eval,
 early stopping, wandb, checkpointing) and thin subclasses per use case:
- DistillationTrainerV0    -> edge model V0 (DeBERTa teacher, different tokenizer)
- TODO: IncrementalDistiller     -> V1, V2, ... VN (rolling previous student as teacher)  [TODO]

 weights are computed from the student's labels in the train dataloader. weighting only affects the (1 - alpha) hard term — with alpha=0.3
 the hard loss is the dominant 0.7 share, so the weighting will have real effect
"""


def distillation_loss(
    student_logits, teacher_logits, labels, T, alpha, class_weights=None, n_old=None
):
    """
    Formal distillation loss following Hinton et al's

    # we add the n_old variable which means the number of n old intents compared to the current version of the model.
    Since we are performing distillation over the old set of intents to preserve the performance of the model on them,
    the teacher only covers the n_old classes.


    """

    if n_old is None:
        kd_student_logits = student_logits  # Using the whole Head
    else:
        kd_student_logits = student_logits[
            :, :n_old
        ]  # a slice of the [0, n_old-1] classes - only performing Knowledge Distillation on this part of the Head

    soft_targets = nn.functional.softmax(
        teacher_logits / T, dim=-1
    )  # revealing dark knowledge with temperature

    hard_loss = F.cross_entropy(student_logits.float(), labels, weight=class_weights)
    # learns new + reinforces the old intents
    # it is needed to align the indices of the old class logits to the current new model, for that since the new intent is
    # only added at the end, 0 to n_old-1 are the same classes in V0 and V1

    soft_prob = nn.functional.log_softmax(kd_student_logits / T, dim=-1)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper
    # "Distilling the knowledge in a neural network" - KL Divergence - how much the student's distribution diverges from the teacher's
    soft_targets_loss = (
        torch.sum(soft_targets * (soft_targets.log() - soft_prob))
        / soft_prob.size()[0]
        * (T**2)
    )

    # soft_targets_loss = F.kl_div(xº
    #     F.log_softmax(student_logits / T, dim=-1),
    #     F.softmax(teacher_logits / T, dim=-1),
    #     reduction="batchmean",
    # ) * (T**2)
    return alpha * soft_targets_loss + (1 - alpha) * hard_loss


class BaseDistillationTrainer:
    """
    Shared knowledge-distillation machinery: optimizer/scheduler/AMP setup, the epoch
    loop, evaluation, early stopping, wandb logging and checkpointing. Mirrors most of
    the training methodology of the directly fine-tuned Trainer class.

    Variant-specific behaviour is delegated to two hooks the subclasses implement:
      ==> _setup_label_space`` -> validate teacher/student head sizes and set
        ``self.num_labels`` (V0 requires an exact match; incremental steps grow it).
      ==> _train_one_epoch the forward/loss step, which differs in the teacher
        tokenizer  and (for incremental) in which logits enter the KD term.
    """

    def __init__(
        self,
        student_model,  # both models need to be passed, inference on teacher to use logits on students for training
        teacher_model,
        student_name,  # Student key name for configuration loading
        teacher_tokenizer,  # needed possible missmatch between tokenizer of teacher-student so possible re-tokenisation
        student_dataloaders,  # the dataloaders with the data to train on
        config,  # The configuration loaded from the config.yaml
        seed=42,
        output_dir=None,  # overrides distill_cfg["output_dir"]; used for per-version incremental paths
    ):
        set_seed(seed)  # reproducibility
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.student_dataloaders = student_dataloaders
        self.train_dataloader = student_dataloaders["train_set"]
        self.eval_dataloader = student_dataloaders["eval_set"]
        self.teacher_tokenizer = teacher_tokenizer
        self.config = config
        self.student_name = student_name
        distill_cfg = config[student_name]["distill"]
        self.epochs = config[student_name]["epochs"]
        weight_scheme = config[self.student_name]["class_weights"]

        # Variant-specific: validate the head sizes and set self.num_labels.
        self._setup_label_space()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(
            False
        )  # No backpropagation needed for the teacher and it saves memory
        self.patience = config[student_name]["patience"]
        self.class_weights = compute_class_weights(
            self.train_dataloader, self.num_labels, weight_scheme, self.device
        )  # this calculates the class weights for an imbalanced dataset.
        self.T = distill_cfg["temperature"]
        self.alpha = distill_cfg["alpha"]
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config[self.student_name]["lr"],
            weight_decay=distill_cfg["decay"],
        )

        self.training_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = (  # Scheduler to warm-up the initial 10% of training epochs
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(
                    self.training_steps * config[self.student_name]["warm-up"]
                ),
                num_training_steps=self.training_steps,
            )
        )
        self.scaler = torch.amp.GradScaler()
        self.history = {
            "train_loss": [],
            "lr": [],
        }
        self.patience_counter = 0
        self.best_macro_f1 = 0.0
        # output_dir (if given) wins, so the incremental run can supply a per-version
        # path (..._v1, ..._v2); V0 passes nothing and keeps the config value.
        self.checkpoint_path = output_dir or distill_cfg["output_dir"]
        os.makedirs(self.checkpoint_path, exist_ok=True)

    # hooks - interfaces
    def _setup_label_space(self):
        """Validate teacher/student head sizes and set ``self.num_labels``."""
        raise NotImplementedError

    def _train_one_epoch(self, epoch):
        """One training epoch; returns the average training loss."""
        raise NotImplementedError

    # shared train/eval of DistilledV0 and DistilledIncremental
    def train(self):
        wandb.init(
            project=f"kd-multilingual-{self.config[self.student_name]['name']}".replace(
                "/", "-"
            ),
            name=self.student_name,
            config=self.config[self.student_name],
        )
        # start training
        for epoch in range(0, self.epochs):
            loss = self._train_one_epoch(epoch)

            metrics = self._eval(epoch)
            last_lr = self.scheduler.get_last_lr()[0]
            self.history["train_loss"].append(loss)
            self.history["lr"].append(last_lr)
            for k, v in metrics.items():
                self.history.setdefault(k, []).append(v)

            wandb.log({"epoch": epoch, "train_loss": loss, "lr": last_lr, **metrics})
            if (
                metrics["eval_macro_f1"] > self.best_macro_f1
            ):  # Use f1 macro for early stopping.
                self.patience_counter = 0
                self.best_macro_f1 = metrics["eval_macro_f1"]
                print("Saving checkpoint")
                # Save the best model INTO the checkpoint dir itself, so it is the version
                # dir the next incremental step loads as its teacher (clean rolling chain).
                self.student_model.save_pretrained(self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        with open(  # saves checkpoints
            os.path.join(self.checkpoint_path, f"history-{self.student_name}.json"), "w"
        ) as f:
            json.dump(self.history, f, indent=2)

        self.student_model.save_pretrained(f"{self.checkpoint_path}_last")
        wandb.finish()

    def _eval(self, epoch):
        self.student_model.eval()
        eval_losses = []
        y_true = []
        y_predicted = []
        y_locales = []  # keep the language for multi-lingual evaluation

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader, desc=f"testing epoch: {epoch}/{self.epochs}"
            ):
                locales = batch.pop("locale")
                utt = batch.pop("utt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.student_model(**batch)
                eval_losses.append(outputs.loss.item())
                y_predicted += outputs.logits.argmax(-1).cpu().tolist()
                y_true += batch["labels"].cpu().tolist()
                y_locales += list(locales)

        avg_loss = sum(eval_losses) / len(eval_losses)
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_predicted)
        loc_arr = np.array(y_locales)
        metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy_score(y_true, y_predicted),
            "eval_macro_f1": f1_score(y_true, y_predicted, average="macro"),
            "eval_weighted_f1": f1_score(y_true, y_predicted, average="weighted"),
        }
        for language in np.unique(loc_arr):
            mask = loc_arr == language
            metrics[f"eval_accuracy_{language}"] = accuracy_score(
                y_true_arr[mask], y_pred_arr[mask]
            )
            metrics[f"eval_macro_f1_{language}"] = f1_score(
                y_true_arr[mask], y_pred_arr[mask], average="macro"
            )
            metrics[f"eval_weighted_f1_{language}"] = f1_score(
                y_true_arr[mask], y_pred_arr[mask], average="weighted"
            )
        return metrics


class DistillationTrainerV0(BaseDistillationTrainer):
    """
    Edge model V0: distil a large teacher ( DeBERTa) into the small student.
    Teacher and student use different tokenizers, so each batch's raw utterances is
    re-tokenised for the teacher, and both share the same label space.
    """

    def _setup_label_space(self):
        student_n = self.student_model.config.num_labels
        teacher_n = self.teacher_model.config.num_labels
        if student_n != teacher_n:
            raise ValueError(
                f"Label-space mismatch: student={student_n}, teacher={teacher_n}. "
                "Distillation requires matching classifier heads."
            )
        self.num_labels = student_n

    def _train_one_epoch(self, epoch):
        training_loss = []
        self.student_model.train()
        for batch in tqdm(
            self.train_dataloader, desc=f"training epoch: {epoch}/{self.epochs}"
        ):
            locales = batch.pop("locale")
            utt = batch.pop("utt")
            # Different tokenizer than students
            teacher_batch = self.teacher_tokenizer(
                utt,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            # moving batches to GPU like student batch
            teacher_batch = {k: v.to(self.device) for k, v in teacher_batch.items()}
            student_batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                # Forward pass of teacher for soft logits distribution
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=teacher_batch["input_ids"],
                        attention_mask=teacher_batch["attention_mask"],
                    )
                # Forward pass of student model for student softmax logits
                student_outputs = self.student_model(**student_batch)
                # loss = outputs.loss
                loss = distillation_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    student_batch["labels"],
                    self.T,
                    self.alpha,
                    self.class_weights,
                )
            # loss = self.criterion(outputs.logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(), max_norm=1.0
            )  # Gradient Clipping: Constraining the gradients during backpropagation to a predefined range
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Log running loss to wandb
            self.scheduler.step()
            wandb.log({"training_loss_step": loss.item()})
            training_loss.append(loss.item())
        return sum(training_loss) / len(
            training_loss
        )  # average training loss across the epoch


# TODO: CHange imports in notebooks and erase this backwards compatibility of both classes
DistillationTrainer = DistillationTrainerV0


def run_distillation_training(
    dataclass: DataClass, teacher_model, teacher_tokenizer, models_keys, config
):
    for key in models_keys:
        student_model, student_tokenizer, _ = build_model(
            config[key]["name"], dataclass.num_labels
        )
        dft_pre_dataloader = dataclass.get_dataloader_data(
            key, student_tokenizer, keep_utt=True
        )
        trainer = DistillationTrainerV0(
            student_model=student_model,
            teacher_model=teacher_model,
            student_name=key,
            teacher_tokenizer=teacher_tokenizer,
            student_dataloaders=dft_pre_dataloader,
            config=config,
        )
        trainer.train()


def run_incremental_step(
    dataclass: DataClass,
    new_intent_name,
    student_key,
    config,
    version,
    K,
    seed=42,
    new_utt=None,
):
    """
    Method that performs a single incremental step of the model with a new intent.
    - It adds the new intent to the dataclass
    - it grows the head of the model with the frozen old teacher weights + randomly initializing the new class
    - It creates the balanced set of data with the new utterances and new intent to pass on to the trainer
    - It trains the new model
    """
    previous_chkpt = _get_incremental_version_dir(
        config, student_key, version - 1
    )  # teacher model directory
    output_directory = _get_incremental_version_dir(
        config, student_key, version
    )  # where the new model will be stored

    teacher_model = AutoModelForSequenceClassification.from_pretrained(previous_chkpt)
    tokenizer = AutoTokenizer.from_pretrained(config[student_key]["name"])
    # first we grow the label space with a new intent, for experimental using to-train set
    dataclass.admit_intent(new_intent_name, new_utt=new_utt)
    # second we perform the head growth of the classifier, always by 1 new intent per incremental iteration
    # n_new is the result of the current num_Labels after admitting it and the previously trained VN model now used as teacher
    n_new = dataclass.num_labels - teacher_model.config.num_labels
    student_model = expand_student_head(
        model=AutoModelForSequenceClassification.from_pretrained(previous_chkpt),
        n_new=n_new,
    )
    # Once we have the student_model with the old weights frozen and the new one initialized
    # we need the dataloader sets that are going to be using for the incremental training.

    dataloader = dataclass.build_incremental_dataloaders(
        student_key, tokenizer, K, seed
    )  # this is the balanced replay buffer

    trainer = IncrementalDistiller(
        student_model=student_model,
        teacher_model=teacher_model,
        student_name=student_key,
        teacher_tokenizer=tokenizer,
        student_dataloaders=dataloader,
        config=config,
        seed=seed,
        output_dir=output_directory,
    )
    trainer.train()
    return output_directory


def _get_incremental_version_dir(config, student_key, version):
    base = config[student_key]["distill"]["output_dir"]
    return f"{base}_v{version}"


class IncrementalDistiller(BaseDistillationTrainer):
    """
    this class inherits from the base distillation trainer and is used for the iteration where
    a knowledge increment occurs, it has to be handled differently than the original student model
    since the teacher model switches to the previous model of the previous iteration, and whilst you are incrementing
    the intent capability of the model, it needs to perform K.D. over the previously known intents to avoid catastrophic forgetting

    """

    def _setup_label_space(self):
        student_n = self.student_model.config.num_labels
        teacher_n = self.teacher_model.config.num_labels
        if student_n <= teacher_n:
            raise ValueError(
                f"Label-space mismatch: student={student_n}, teacher={teacher_n}. "
                "Incremental step requires the student head to have grown and therefore be bigger than the teacher"
            )

        self.num_labels = student_n  # This here is the full head: old intents + the fresh newly added one
        self.n_old = teacher_n  # the number of n_old intents correspond to the intents of the teacher model
        self.n_new = (
            student_n - teacher_n
        )  # How many intents were admitted in this step !! This means that it is not restricted to a single new intent added per iteration

    def _train_one_epoch(self, epoch):
        training_loss = []
        self.student_model.train()
        for batch in tqdm(
            self.train_dataloader, desc=f"training epoch: {epoch}/{self.epochs}"
        ):
            locales = batch.pop("locale")
            utt = batch.pop("utt")
            # here the tokenizer is not needed, since both teacher and student share the same.
            # moving batches to GPU like student batch
            student_batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                # Forward pass of teacher for soft logits distribution
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=student_batch["input_ids"],
                        attention_mask=student_batch["attention_mask"],
                    )
                # Forward pass of student model for student softmax logits
                student_outputs = self.student_model(**student_batch)
                # loss = outputs.loss
                loss = distillation_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    student_batch["labels"],
                    self.T,
                    self.alpha,
                    self.class_weights,
                    n_old=self.n_old,  # we pass here the teacher's old intents positions
                )
            # loss = self.criterion(outputs.logits, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(), max_norm=1.0
            )  # Gradient Clipping: Constraining the gradients during backpropagation to a predefined range
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Log running loss to wandb
            self.scheduler.step()
            wandb.log({"training_loss_step": loss.item()})
            training_loss.append(loss.item())
        return sum(training_loss) / len(
            training_loss
        )  # average training loss across the epoch


### Main to test the class before starting the proper training on google collab.

# if __name__ == "__main__":
