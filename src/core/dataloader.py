from datasets import load_dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader


def load_massive_dataset(config, languages, sets_names):
    """
    Function that loads the MASSIVE dataset and take the different splits (Train/validation/test)
    """
    dataset = config["dataset"]["name"]
    columns = ["id", "utt", "intent", "locale"]
    train_df = []
    eval_df = []
    test_df = []
    for lan in languages:
        ds = load_dataset(dataset, lan)  # trust_remote_code=True
        train_df.append(ds["train"].select_columns(columns))
        eval_df.append(ds["validation"].select_columns(columns))
        test_df.append(ds["test"].select_columns(columns))
    dataset_splits = {
        sets_names[0]: concatenate_datasets(train_df),
        sets_names[1]: concatenate_datasets(test_df),
        sets_names[2]: concatenate_datasets(eval_df),
    }
    return DatasetDict(dataset_splits)


def label_map(data):
    """
    Function to map the id's to the string intent and vice-versa
    """
    id2intent = {i: name for i, name in enumerate(data.features["intent"].names)}
    intent2id = {name: i for i, name in enumerate(data.features["intent"].names)}
    return id2intent, intent2id


def fit_tokenizer(dataset, tokenizer, split_names, max_length=128, keep_utt=False):
    """
    This function fits a specific tokenizer to the dataset. Since Each model depends on a certain tokenization strategy
    XLM-Roberta uses SentecePiece, MiniLM uses WordPiece, DistilmBERT miltilingual uses WordPiece

    """

    def tokenize(batch):
        # Pading: to have matching length tokenized sequences
        # Truncation to have a fixed  sequence lenght => max_length
        t = tokenizer(
            batch["utt"], truncation=True, padding="max_length", max_length=max_length
        )

        return t

    datasets_tokenized = {}
    # The map() method works by applying a function on each element of the dataset, so let’s define a function that tokenizes our inputs
    for s in split_names:
        if keep_utt:  # keep the utterance like locale to use for the different tokenizers in the distillation loop.
            split_tok = dataset[s].map(tokenize, batched=True, remove_columns=["id"])
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


def feed_dataloader(dataset_tokenized, sets_names, batch_size=16):
    """
    Once the dataset is tokenized it needs to be transformed to  DataLoader format for training purposes with PyTorch
    """
    dataloaders = {}
    for s in sets_names:
        dataloaders[s] = DataLoader(
            dataset_tokenized[s],
            shuffle=(s == sets_names[0]),  # only shuffle train the rest is not
            batch_size=batch_size,
            # collate_fn=data_collator,
        )
    return dataloaders
