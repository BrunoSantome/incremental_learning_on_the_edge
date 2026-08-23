import os
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
from torch.utils.data import DataLoader
from .configuration import load_config, set_seed
from .constants import constants
from .registry import IntentRegistry
import random
from collections import defaultdict


class DataClass:
    def __init__(self):
        self.config = load_config()
        self.languages = self.config[constants.DATASET][constants.LANG]
        self.sets_names = self.config[constants.SPLIT_NAME]
        self.label_col = self.config[constants.DATASET][constants.LABEL]
        self.dataset_name = self.config[constants.DATASET][constants.NAME]
        self.dataset_col_selected = self.config[constants.DATASET][constants.COLUMNS]
        self.registry_path = self._resolve_path(
            self.config[constants.REGISTRY_PATH]
        )  # ADD TO CONFIG

        self.dataset_dict = self.load_massive_dataset()
        # Map raw MASSIVE intent ids -> names, then build the persistent, global
        # intent registry from the pretrain intents (indices 0..N-1).
        names = self.dataset_dict[self.sets_names[0]].features[self.label_col].names
        self.id2name = {i: name for i, name in enumerate(names)}
        pretrain_names = [
            self.id2name[i]
            for i in self.config[constants.DATASET][constants.PRE_TRAIN_INTENTS]
        ]
        self.registry = IntentRegistry.get_or_create(pretrain_names, self.registry_path)
        self.dataset_pretraining, self.dataset_totrain = self.split_by_intent_sets()

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, path)

    @property
    def num_labels(self):
        """Number of classes in the current (pretrain) phase, from the registry."""
        return self.registry.num_intents

    @property
    def id2intent(self):
        """Index -> intent name, for readable evaluation / forgetting reports"""
        return self.registry.id2intent

    def load_massive_dataset(self):
        """
        Function that loads the MASSIVE dataset and take the different splits (Train/validation/test)
        """
        train_df, eval_df, test_df = [], [], []
        for lan in self.languages:
            ds = load_dataset(self.dataset_name, lan)  # trust_remote_code=True
            train_df.append(
                ds[constants.TRAIN].select_columns(self.dataset_col_selected)
            )
            eval_df.append(ds[constants.EVAL].select_columns(self.dataset_col_selected))
            test_df.append(ds[constants.TEST].select_columns(self.dataset_col_selected))
        dataset_splits = {
            self.sets_names[0]: concatenate_datasets(train_df),
            self.sets_names[1]: concatenate_datasets(test_df),
            self.sets_names[2]: concatenate_datasets(eval_df),
        }
        return DatasetDict(dataset_splits)

    def split_by_intent_sets(self):
        """
        Splits a DatasetDict into pre-training intents and reserved (to-learn) intents.

        The pretrain split has its labels remapped from the raw MASSIVE ids to the
        contiguous registry indices (0..N-1) so the classifier head is sized to the
        intents that actually appear.
        """
        pretrain_ids = set(self.config["dataset"]["pretrain_intents"])
        reserve_ids = set(self.config["dataset"]["reserve_intents"])

        pretrain_ds = self.dataset_dict.filter(
            lambda ex: ex[self.label_col] in pretrain_ids
        )
        reserve_ds = self.dataset_dict.filter(
            lambda ex: ex[self.label_col] in reserve_ids
        )
        pretrain_ds = self._remap_to_registry(pretrain_ds)
        self._validate_labels(pretrain_ds)
        reserve_ds = self._remap_to_names(reserve_ds)
        return pretrain_ds, reserve_ds

    def _remap_to_registry(self, dataset_dict):
        """Rewrite raw MASSIVE intent ids to contiguous registry indices."""

        def remap(ex):
            ex[self.label_col] = self.registry[self.id2name[ex[self.label_col]]]
            return ex

        dataset_dict = dataset_dict.map(remap)
        # Drop the stale ClassLabel metadata (its 60 names no longer match the values).
        dataset_dict = dataset_dict.cast_column(self.label_col, Value("int64"))
        return dataset_dict

    def _remap_to_names(self, dataset_dict):
        """
        These intents have noregistry index until they are learned, so a raw numeric id here would be a
        The name is resolved to an index via registry.add_intent(name) at admission.
        """

        # Cast off ClassLabel first: otherwise map() returning a name string would be
        # re-encoded by ClassLabel straight back to the raw id, undoing the mapping.
        dataset_dict = dataset_dict.cast_column(self.label_col, Value("int64"))

        def remap(ex):
            ex[self.label_col] = self.id2name[ex[self.label_col]]
            return ex

        dataset_dict = dataset_dict.map(remap)
        dataset_dict = dataset_dict.cast_column(self.label_col, Value("string"))
        return dataset_dict

    def admit_intent(self, new_intent_name, new_utt=None):
        """
        Add a new intent into the current pretrained_dataset.
        it assings an integer that matches the new intent label
        it changes the string of the intent with the number
        it adds the new set of utterances into the pretrained_dataset
        and removes it from the to_train in the experimental phase,
        this changes when an LLM generates it
        """
        new_idx = self.registry.add_intent(
            new_intent_name
        )  # persists the name into the registry maping it to a new integer

        merged = {}
        for split in self.sets_names:
            if new_utt is None:  # Experimental with original data for incremental step
                rows = self.dataset_totrain[split].filter(
                    lambda ex: ex[self.label_col] == new_intent_name
                )
                self.dataset_totrain[split] = self.dataset_totrain[split].filter(
                    lambda ex: ex[self.label_col] != new_intent_name
                )
            else:  # TODO: LLM step: injected utterances for this split
                rows = new_utt[split]

            assert len(rows) > 0, (
                f"no rows for intent '{new_intent_name}' in split '{split}'"
            )  # make sure of no empty rows to not move on.

            rows = self._relabel_to_index(rows, new_idx)  # name-string -> int64 index

            # merge into the known set, and drop the intent from the reserve to-train set
            merged[split] = concatenate_datasets(
                [self.dataset_pretraining[split], rows]
            )

        self.dataset_pretraining = DatasetDict(merged)
        return new_idx

    def _relabel_to_index(self, dataset, idx):
        """
        Set every row's label to idx and make the column int64, so the rows concatenate
        with dataset_pretraining.
        """
        dataset = dataset.map(
            lambda ex: {self.label_col: idx}
        )  # write the int into the col
        dataset = dataset.cast_column(self.label_col, Value("int64"))  # pin the type
        return dataset

    def _balanced_sample(self, dataset, K, seed=42):
        """At most K examples per intent, seeded so the buffer is reproducible."""
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for i, y in enumerate(dataset[self.label_col]):
            by_class[y].append(i)

        idx = []
        for c in sorted(by_class):
            rows = by_class[c]
            rng.shuffle(rows)
            idx += rows[:K]

        return dataset.select(idx)

    def _validate_labels(self, dataset_dict):
        """Sanity check: remapped labels must fill exactly range(num_labels)."""
        expected = set(range(self.num_labels))
        for split in self.sets_names:
            found = set(dataset_dict[split].unique(self.label_col))
            assert found <= expected, (
                f"{split}: labels {found - expected} outside 0..{self.num_labels - 1}"
            )

    def label_map(self, data):
        """
        Function to map the id's to the string intent and vice-versa
        """
        id2intent = {
            i: name for i, name in enumerate(data.features[constants.INTENT].names)
        }
        intent2id = {
            name: i for i, name in enumerate(data.features[constants.INTENT].names)
        }
        return id2intent, intent2id

    def fit_tokenizer(self, dataset, tokenizer, max_length=128, keep_utt=False):
        """
        This function fits a specific tokenizer to the dataset. Since Each model depends on a certain tokenization strategy
        XLM-Roberta uses SentecePiece, MiniLM uses WordPiece, DistilmBERT miltilingual uses WordPiece

        """

        def tokenize(batch):
            # Pading: to have matching length tokenized sequences
            # Truncation to have a fixed  sequence lenght => max_length
            t = tokenizer(
                batch["utt"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

            return t

        datasets_tokenized = {}
        # The map() method works by applying a function on each element of the dataset, so let’s define a function that tokenizes our inputs
        for s in self.sets_names:
            if keep_utt:  # keep the utterance like  to use for localethe different tokenizers in the distillation loop.
                split_tok = dataset[s].map(
                    tokenize, batched=True, remove_columns=["id"]
                )
            else:
                split_tok = dataset[s].map(
                    tokenize, batched=True, remove_columns=["id", "utt"]
                )  # callback function with the batch of data.
            split_tok = split_tok.rename_column("intent", "labels")
            # important to keep the labels too, to have the utterance with its label

            # Attetion mask tells which tokens are padding and which not
            tensor_cols = [
                col
                for col in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                if col in split_tok.column_names
            ]
            split_tok.set_format(
                type="torch",
                columns=tensor_cols,
                output_all_columns=True,  # to not ommit locale
            )
            datasets_tokenized[s] = split_tok
        return datasets_tokenized

    def feed_dataloader(self, dataset_tokenized, model_key):
        """
        Once the dataset is tokenized it needs to be transformed to DataLoader format for training purposes with PyTorch
        """
        batch_size = self.config[model_key]["batch_size"]
        dataloaders = {}
        for s in self.sets_names:
            dataloaders[s] = DataLoader(
                dataset_tokenized[s],
                shuffle=(s == self.sets_names[0]),  # only shuffle train the rest is not
                batch_size=batch_size,
                # collate_fn=data_collator,
            )
        return dataloaders

    def get_dataloader_data(self, model_key, tokenizer, pre_data=True, keep_utt=False):
        if pre_data:
            dft_pre = self.fit_tokenizer(
                self.dataset_pretraining, tokenizer, keep_utt=keep_utt
            )
            return self.feed_dataloader(dft_pre, model_key)
        if not pre_data:
            dft_post = self.fit_tokenizer(
                self.dataset_totrain, tokenizer, keep_utt=keep_utt
            )
            return self.feed_dataloader(dft_post, model_key)

    def build_incremental_dataloaders(self, student_key, tokenizer, K, seed=42):
        """
        This function creates the replay buffer data to re-train a model with previous utterances
        matching K number so it is balanced across iterations, used for avoiding Catastrophical forgetting

        """
        train_split, test_split, eval_split = self.sets_names

        incremental = DatasetDict(
            {
                train_split: self._balanced_sample(
                    self.dataset_pretraining[train_split], K, seed
                ),
                test_split: self.dataset_pretraining[test_split],
                eval_split: self.dataset_pretraining[eval_split],
            }
        )
        tokenized = self.fit_tokenizer(incremental, tokenizer, keep_utt=True)
        return self.feed_dataloader(tokenized, student_key)


# if __name__ == "__main__":
#     dataset = DataClass()
#     print(dataset.dataset_pretraining)
#     print(dataset.dataset_totrain)
