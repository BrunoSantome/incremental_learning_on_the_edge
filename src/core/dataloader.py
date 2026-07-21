from datasets import load_dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from configuration import load_config, set_seed
from constants import constants


class DataClass:
    def __init__(self):
        self.config = load_config()
        self.languages = self.config[constants.DATASET][constants.LANG]
        self.sets_names = self.config[constants.SPLIT_NAME]
        self.label_col = self.config[constants.DATASET][constants.LABEL]
        self.dataset_name = self.config[constants.DATASET][constants.NAME]
        self.dataset_col_selected = self.config[constants.DATASET][constants.COLUMNS]
        self.dataset_dict = self.load_massive_dataset()
        self.dataset_pretraining, self.dataset_totrain = self.split_by_intent_sets()

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
        """
        pretrain_ids = set(self.config["dataset"]["pretrain_intents"])
        reserve_ids = set(self.config["dataset"]["reserve_intents"])

        pretrain_ds = self.dataset_dict.filter(
            lambda ex: ex[self.label_col] in pretrain_ids
        )
        reserve_ds = self.dataset_dict.filter(
            lambda ex: ex[self.label_col] in reserve_ids
        )
        return pretrain_ds, reserve_ds

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

    def get_dataloader_data(self, model_key, tokenizer, pre_data=True):
        if pre_data:
            dft_pre = self.fit_tokenizer(self.dataset_pretraining, tokenizer)
            return self.feed_dataloader(dft_pre, model_key)
        if not pre_data:
            dft_post = self.fit_tokenizer(self.dataset_totrain, tokenizer)
            return self.feed_dataloader(dft_post, model_key)


if __name__ == "__main__":
    dataset = DataClass()
    print(dataset.dataset_pretraining)
    print(dataset.dataset_totrain)
