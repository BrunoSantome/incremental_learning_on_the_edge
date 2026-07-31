from .utils import build_model, compute_class_weights
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
 
 Check if worth changing the distillation class entirely to another for the incremental distillation or 
 change the class so it can be used in both use cases: Distillatin of edge model V0 and incremental V1,V2,V3 


 weights are computed from the student's labels in the train dataloader. weighting only affects the (1 - alpha) hard term — with alpha=0.3
 the hard loss is the dominant 0.7 share, so the weighting will have real effect
"""


def distillation_loss(
    student_logits, teacher_logits, labels, T, alpha, class_weights=None
):
    """
    Formal distillation loss following Hinton et al's

    """

    soft_targets = nn.functional.softmax(
        teacher_logits / T, dim=-1
    )  # revealing dark knowledge with temperature
    soft_prob = nn.functional.log_softmax(
        student_logits / T, dim=-1
    )  # students logits to match the same "temperature scale"
    hard_loss = F.cross_entropy(student_logits.float(), labels, weight=class_weights)
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


class DistillationTrainer:
    """
    Distillation Trainer class which performs knowledge distillation using a teacher's inference on a student model.
    The class mimics most of the training methodologies used in the directly fine-tuned Trainer class.

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
    ):
        set_seed(seed)  # reproducibility
        self.student_model = student_model
        self.teacher_model = teacher_model

        student_n = self.student_model.config.num_labels
        teacher_n = self.teacher_model.config.num_labels
        if student_n != teacher_n:
            raise ValueError(
                f"Label-space mismatch: student={student_n}, teacher={teacher_n}. "
                "Distillation requires matching classifier heads."
            )
        self.num_labels = student_n
        self.student_dataloaders = student_dataloaders
        self.train_dataloader = student_dataloaders["train_set"]
        self.eval_dataloader = student_dataloaders["eval_set"]
        self.teacher_tokenizer = teacher_tokenizer
        self.config = config
        self.student_name = student_name
        self.epochs = config[student_name]["epochs"]
        weight_scheme = config[self.student_name]["class_weights"]
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
        class_weights = compute_class_weights(
            self.train_dataloader, self.num_labels, weight_scheme, self.device
        )  # this calculates the class weights for an imbalanced dataset.
        self.T = config[student_name]["temperature"]
        self.alpha = config[student_name]["alpha"]
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config[self.student_name]["lr"],
            weight_decay=config[self.student_name]["decay"],
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
        self.checkpoint_path = config[self.student_name]["output_dir"]
        os.makedirs(self.checkpoint_path, exist_ok=True)

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

    def train(self):
        wandb.init(
            project=f"kd-multilingual-{self.config[self.student_name]['name']}".replace(
                "/", "-"
            ),
            name=self.student_name,
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


def run_distillation_training(
    dataclass, teacher_model, teacher_tokenizer, models_keys, config
):
    for key in models_keys:
        student_model, student_tokenizer, _ = build_model(
            config[key]["name"], dataclass.num_labels
        )
        dft_pre_dataloader = dataclass.get_dataloader_data(
            key, student_tokenizer, keep_utt=True
        )
        trainer = DistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            student_name=key,
            teacher_tokenizer=teacher_tokenizer,
            student_dataloaders=dft_pre_dataloader,
            config=config,
        )
        trainer.train()


### Main to test the class before starting the proper training on google collab.

# if __name__ == "__main__":
