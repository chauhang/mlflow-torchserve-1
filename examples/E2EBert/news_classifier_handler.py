import json
import logging
import os
import numpy as np
import torch
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler
from transformers import BertForSequenceClassification, BertConfig
from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, \
    remove_interpretable_embedding_layer
import torch.nn as nn
import json
import logging
import os
import numpy as np
import torch
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler
from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from news_classifier import BertNewsClassifier
import torch.nn.functional as F
import torch.nn as nn
import torch.nn as nn

from news_classifier import BertNewsClassifier

from wrapper import AGNewsmodelWrapper

logger = logging.getLogger(__name__)


class NewsClassifierHandler(BaseHandler):
    """
    NewsClassifierHandler class. This handler takes a review / sentence
    and returns the label as either world / sports / business /sci-tech
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.class_mapping_file = None
        self.VOCAB_FILE = None

    def initialize(self, ctx):
        """
        First try to load torchscript else load eager mode state_dict based model

        :param ctx: System properties
        """

        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "state_dict.pth")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "news_classifier.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")
        self.VOCAB_FILE = os.path.join(model_dir, "bert_base_uncased_vocab.txt")
        if not os.path.isfile(self.VOCAB_FILE):
            raise RuntimeError("Missing the vocab file")

        self.class_mapping_file = os.path.join(model_dir, "class_mapping.json")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = BertNewsClassifier()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True

    def preprocess(self, data):
        """
        Receives text in form of json and converts it into an encoding for the inference stage

        :param data: Input to be passed through the layers for prediction

        :return: output - preprocessed encoding
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        self.text = text.decode("utf-8")

        self.tokenizer = BertTokenizer(self.VOCAB_FILE)
        self.input_ids = torch.tensor([self.tokenizer.encode(self.text, add_special_tokens=True)])
        print("Shape of encoding input_ids", self.input_ids.shape)
        return self.input_ids

    def inference(self, input_ids):
        """
        Predict the class  for a review / sentence whether it is belong to world / sports / business /sci-tech
        :param encoding: Input encoding to be passed through the layers for prediction

        :return: output - predicted output
        """
        inputs = self.input_ids.to(self.device)
        self.outputs = self.model.forward(inputs)
        self.out = np.argmax(self.outputs.cpu().detach())
        return [self.out.item()]

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """
        if os.path.exists(self.class_mapping_file):
            with open(self.class_mapping_file) as json_file:
                data = json.load(json_file)
            inference_output = json.dumps(data[str(inference_output[0])])
            return [inference_output]

        return inference_output


    def explain_initialize(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_dir = os.getcwd()
        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "state_dict.pth")
        # Read model definition file
        VOCAB_FILE = os.path.join(model_dir, "bert_base_uncased_vocab.txt")
        if not os.path.isfile(VOCAB_FILE):
            raise RuntimeError("Missing the vocab file")

        class_mapping_file = os.path.join(model_dir, "class_mapping.json")
        state_dict = torch.load(model_pt_path, map_location=device)
        model = BertNewsClassifier()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        tokenizer = BertTokenizer(VOCAB_FILE)

    def add_attributions_to_visualizer(self, attributions, tokens, pred_prob, pred_class, true_class,
                                       attr_class, delta, vis_data_records):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        # storing couple samples in an array for visualization purposes
        vis_data_records.append(visualization.VisualizationDataRecord(
            attributions,
            pred_prob,
            pred_class,
            true_class,
            attr_class,
            attributions.sum(),
            tokens,
            delta))

    def score_func(self, o):
        output = F.softmax(o, dim=1)
        pre_pro = np.max(output.detach().numpy())
        return pre_pro

    def explain_handle(self, model_wraper, text):
        """Captum explanations handler
        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request
        Returns:
            dict : A dictionary response with the explanations response."""
        vis_data_records_base = []
        model_wrapper = AGNewsmodelWrapper(self.model)
        tokenizer = BertTokenizer(self.VOCAB_FILE)
        model_wrapper.eval()
        model_wrapper.zero_grad()
        input_ids = torch.tensor([tokenizer.encode(self.text, add_special_tokens=True)])
        input_embedding_test = model_wrapper.model.bert_model.embeddings(input_ids)
        preds = model_wrapper(input_embedding_test)
        out = np.argmax(preds.cpu().detach(), axis=1)
        out = (out.item())
        ig_1 = IntegratedGradients(model_wrapper)
        attributions, delta = ig_1.attribute(input_embedding_test, n_steps=500,
                                             return_convergence_delta=True, target=1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())
        attr = attributions.detach().numpy()
        importances = np.mean(attr, axis=0)
        feature_imp_dict = {}
        for i in range(len(tokens)):
            feature_imp_dict[str(tokens[i])] = np.mean(importances[i])
        self.add_attributions_to_visualizer(attributions, tokens, self.score_func(preds), out, 2, 1,
                                       delta, vis_data_records_base)
        return [feature_imp_dict]



