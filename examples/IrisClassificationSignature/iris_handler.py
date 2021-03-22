import ast
import logging

import numpy as np
import torch
import os
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class IRISClassifierHandler(BaseHandler):
    """
    IRISClassifier handler class. This handler takes an input tensor and
    output the type of iris based on the input
    """

    def __init__(self):
        super(IRISClassifierHandler, self).__init__()

    def preprocess(self, data):
        """
        preprocessing step - Reads the input array and converts it to tensor

        :param data: Input to be passed through the layers for prediction

        :return: output - Preprocessed input
        """

        # input_data_str = data[0].get("data")
        # if input_data_str is None:
        #     input_data_str = data[0].get("body")
        #
        # input_data = input_data_str.decode("utf-8")
        # input_tensor = torch.Tensor(ast.literal_eval(input_data))
        return data

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
        model_pt_path = os.path.join(model_dir, "model.pth")

        self.model = torch.load(model_pt_path, map_location=self.device)

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        self.initialized = True

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """

        predicted_idx = str(np.argmax(inference_output.cpu().detach().numpy()))

        if self.mapping:
            return [self.mapping[str(predicted_idx)]]
        return [predicted_idx]


_service = IRISClassifierHandler()
