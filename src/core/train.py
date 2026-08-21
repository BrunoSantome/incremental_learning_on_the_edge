from .utils import build_model, compute_class_weights
from .configuration import set_seed, load_config
from .dataloader import DataClass
from transformers import get_linear_schedule_with_warmup
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
import numpy as np
import json


class Trainer:
    """
    Trainer class to directly fine-tune all the models.
    Fine-tune the teacher models to use it for knowledge distillation and fine-tune the students for a direct comparison.
    """

    def __init__(
        self,
        model,  # The model to be trained
        model_key,  # the key of the model for config loading purposes
        dataloaders,  # the data to train on
        config,  # the configuration loaded from config.yaml
        seed=42,
    ):  # the seed for reproducibility
        set_seed(seed)
        self.model = model
        self.num_labels = self.model.config.num_labels
        self.dataloaders_dict = dataloaders
        self.train_dataloader = dataloaders["train_set"]
        self.eval_dataloader = dataloaders["eval_set"]
        self.config = config
        self.model_key = model_key
        self.epochs = config[model_key]["epochs"]
        weight_scheme = config[model_key]["class_weights"]

        if torch.cuda.is_available():  # check if cuda is available
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)  # moves model to GPU
        self.patience = config[
            model_key
        ][
            "patience"
        ]  # patience for early stopping, Number of epochs to wait before stopping if the results are not improved.
        class_weights = compute_class_weights(
            self.train_dataloader, self.num_labels, weight_scheme, self.device
        )  # this calculates the class weights for an imbalanced dataset.
        self.optimizer = torch.optim.AdamW(  # Adamw optimzer loading the decay-rate
            self.model.parameters(),
            lr=config[self.model_key]["lr"],
            weight_decay=config[self.model_key]["decay"],
        )
        label_smoothing = config[model_key].get("label_smoothing", 0.0)
        self.criterion_smoothing = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
        # self.criterion = torch.nn.CrossEntropyLoss(
        #     weight=class_weights
        # )  # class weighted CE
        self.eval_criterion = (
            torch.nn.CrossEntropyLoss()
        )  # no weights in the evaluation loop
        self.training_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = (
            get_linear_schedule_with_warmup(  # Scheduler with warm-up steps
                self.optimizer,
                num_warmup_steps=int(
                    self.training_steps * config[self.model_key]["warm-up"]
                ),
                num_training_steps=self.training_steps,
            )
        )
        self.scaler = torch.amp.GradScaler()
        self.history = {
            "train_loss": [],
            "lr": [],
        }
        self.best_eval_loss = float("inf")
        self.patience_counter = 0
        self.best_macro_f1 = 0.0
        self.checkpoint_path = config[self.model_key]["output_dir"]
        os.makedirs(self.checkpoint_path, exist_ok=True)
        # ~need to init list of loss and metrics? create evaluation ?

    def _train_one_epoch(self, epoch):
        training_loss = []
        self.model.train()
        for batch in tqdm(
            self.train_dataloader, desc=f"training epoch: {epoch}/{self.epochs}"
        ):
            _ = batch.pop("locale")
            batch = {
                k: v.to(self.device) for k, v in batch.items()
            }  # pass to the device all tensors.
            # batch = {
            #     "input_ids": batch["input_ids"].to(self.device),
            #     "attention_mask": batch["attention_mask"].to(self.device),
            #     "labels": batch["labels"].to(self.device),
            # }
            labels = batch.pop("labels")
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(**batch)
                loss = self.criterion_smoothing(
                    outputs.logits.float(),
                    labels,  # loss in fp32 under amp, avoiding computer cross entropy on fp16 logits under autocast
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
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
            project=f"Teacher_model-{self.config[self.model_key]['name']}".replace(
                "/", "-"
            ),
            name=self.model_key,
            config=self.config[self.model_key],
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
            if metrics["eval_macro_f1"] > self.best_macro_f1:
                self.patience_counter = 0
                self.best_macro_f1 = metrics["eval_macro_f1"]
                print("Saving checkpoint")
                self.model.save_pretrained(self.checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        with open(
            os.path.join(self.checkpoint_path, f"history-{self.model_key}.json"), "w"
        ) as f:
            json.dump(self.history, f, indent=2)

        self.model.save_pretrained(f"{self.checkpoint_path}_direct_last")
        wandb.finish()

    def _eval(self, epoch):
        self.model.eval()
        eval_losses = []
        y_true = []
        y_predicted = []
        y_locales = []  # keep the language for multi-lingual evaluation

        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader, desc=f"testing epoch: {epoch}/{self.epochs}"
            ):
                locales = batch.pop("locale")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(**batch)
                eval_losses.append(
                    self.eval_criterion(outputs.logits.float(), labels).item()
                )
                y_predicted += outputs.logits.argmax(-1).cpu().tolist()
                y_true += labels.cpu().tolist()
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


### Main to test the class before starting the proper training on google collab.

# if __name__ == "__main__":
#     config = load_config()
#     dataclass = DataClass()
#     model_key = "student1"
#     set_seed(42)
#     model, tokenizer, _ = build_model(config[model_key]["name"], dataclass.num_labels)
#     dft_pre_dataloader = dataclass.get_dataloader_data(model_key, tokenizer)
#     trainer = Trainer(model, model_key, dft_pre_dataloader, config)
#     trainer.train()
