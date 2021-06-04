import os
import pandas as pd
import requests
import torch
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchtext.datasets as td
from transformers import BertTokenizer, BertModel


class AGNewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer

        :param reviews: AG news text
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text

        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe

        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item

        :param item: Index of sample review

        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.reviews[item])
        target = self.targets[item]
        encoded = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertDataHandler:
    tokenizer = None
    NUM_SAMPLES_COUNT = 1500
    VOCAB_FILE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    VOCAB_FILE = "bert_base_uncased_vocab.txt"
    MAX_LEN = 100
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    @staticmethod
    def setup():
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        # reading  the input
        td.AG_NEWS(root="data", split=("train", "test"))
        extracted_files = os.listdir("data")

        train_csv_path = None
        for fname in extracted_files:
            if fname.endswith("train.csv"):
                train_csv_path = os.path.join(os.getcwd(), "data", fname)

        df = pd.read_csv(train_csv_path)

        df.columns = ["label", "title", "description"]
        df.sample(frac=1)
        df = df.iloc[: BertDataHandler.NUM_SAMPLES_COUNT]

        df["label"] = df.label.apply(BertDataHandler.process_label)

        if not os.path.isfile(BertDataHandler.VOCAB_FILE):
            filePointer = requests.get(BertDataHandler.VOCAB_FILE_URL, allow_redirects=True)
            if filePointer.ok:
                with open(BertDataHandler.VOCAB_FILE, "wb") as f:
                    f.write(filePointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")

        BertDataHandler.tokenizer = BertTokenizer(BertDataHandler.VOCAB_FILE)

        for param in BertDataHandler.bert_model.parameters():
            param.requires_grad = False

        RANDOM_SEED = 42
        seed_everything(RANDOM_SEED)

        df_train, df_test = train_test_split(
            df, test_size=0.1, random_state=RANDOM_SEED, stratify=df["label"]
        )

        return BertDataHandler.create_data_loader(
            df_test, BertDataHandler.tokenizer, BertDataHandler.MAX_LEN, 4
        )

    @staticmethod
    def create_data_loader(df, tokenizer, max_len, batch_size):
        """
        Generic data loader function

        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training

        :return: Returns the constructed dataloader
        """
        ds = AGNewsDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=batch_size, num_workers=3
        )

    @staticmethod
    def get_batch_data():
        from captum.insights import Batch
        dataloader = iter(BertDataHandler.setup())
        while True:
            inp_data = next(dataloader)
            yield Batch(inputs=inp_data['input_ids'],
                        labels=inp_data['targets']
                        )

    @staticmethod
    def visualization(input_ids):
        tokens_test = BertDataHandler.tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())
        tokens_test = [i for i in tokens_test if i not in ["[CLS]", "[PAD]", "[SEP]"]]
        return tokens_test

    @staticmethod
    def transform(input):
        input = input.unsqueeze(0)
        input_embedding_test = BertDataHandler.bert_model.embeddings(input)
        return input_embedding_test.squeeze(0)

    @staticmethod
    def baseline_func(input):
        return input * 0
