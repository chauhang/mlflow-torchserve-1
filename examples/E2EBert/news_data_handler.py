import os
import requests
from transformers import BertTokenizer, BertModel


class BertDataHandler:
    NUM_SAMPLES_COUNT = 1500
    VOCAB_FILE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
    VOCAB_FILE = "bert_base_uncased_vocab.txt"
    MAX_LEN = 100
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
    bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if not os.path.isfile(VOCAB_FILE):
        filePointer = requests.get(VOCAB_FILE_URL, allow_redirects=True)
        if filePointer.ok:
            with open(VOCAB_FILE, "wb") as f:
                f.write(filePointer.content)
        else:
            raise RuntimeError("Error in fetching the vocab file")

    tokenizer = BertTokenizer(VOCAB_FILE)

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
