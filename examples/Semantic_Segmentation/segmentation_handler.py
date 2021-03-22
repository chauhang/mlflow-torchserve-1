import json
import os
import torch
from model import FCNImageSegmenter
import logging
from PIL import Image
from ts.torch_handler.image_segmenter import ImageSegmenter
from ts.torch_handler.base_handler import BaseHandler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from io import BytesIO
import mlflow
import torch
import ast
import requests
from torchvision import models
from torchvision import transforms
from captum.attr import visualization as viz
from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
from segmentation_model import preprocessing, get_classes, agg_segmentation_wrapper

logger = logging.getLogger(__name__)


class Sementic_Segmentation(BaseHandler):
    def __init__(self):
        super(Sementic_Segmentation, self).__init__()
        self.initialized = False
        self.preproc_img = None
        self.normalized_inp = None
        self.pred_label_idx = None
        self.predictions = None
        self.output_path = None

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        logger.info("[INFO] Model dir is {}".format(model_dir))
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )

        self.fcn = FCNImageSegmenter()
        self.fcn.load_state_dict(torch.load(model_pt_path))
        self.fcn.to(self.device)
        self.fcn.eval()
        logger.info("[INFO] Segmentation model loaded successfully")

        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            logger.info("[INFO] Mapping file present")
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:

            logger.warning("Missing the index_to_name.json file.")

        self.lgc = LayerGradCam(self.agg_segmentation_wrapper, self.fcn.backbone.layer4[2].conv3)
        self.fa = FeatureAblation(self.agg_segmentation_wrapper)
        self.initialized = True

    def agg_segmentation_wrapper(self, inp):
        model_out = self.fcn(inp)["out"]
        self.out_max = torch.argmax(model_out, dim=1, keepdim=True)
        selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, self.out_max, 1)
        return (model_out * selected_inds).sum(dim=(2, 3))

    def preprocess(self, data):
        data = data[0]["body"].decode("utf-8")
        data = ast.literal_eval(data)
        input_file_path = data["input_file_path"][0]
        self.output_path = data["output_path"]
        response = requests.get(input_file_path)
        img = Image.open(BytesIO(response.content))
        print("[INFO] Image downloaded successfully: ")
        preprocessing_ = transforms.Compose([transforms.Resize(640), transforms.ToTensor()])
        normalize_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc_img = preprocessing_(img)
        self.normalized_inp = normalize_(self.preproc_img).unsqueeze(0).to(self.device)

        return (self.preproc_img, self.normalized_inp)

    def inference(self, normalized_inp):
        self.normalized_inp.requires_grad = True
        pred_label_idx = torch.nonzero(self.agg_segmentation_wrapper, as_tuple=True)[1]
        print("[INFO] Predictions: ", pred_label_idx, type(pred_label_idx))
        return [self.pred_label_idx]

    def postprocess(self, pred_label_idx):
        return pred_label_idx

    def get_insights(self, normalized_inp, target):
        lgc = LayerGradCam(self.agg_segmentation_wrapper, self.fcn.backbone.layer4[2].conv3)
        gc_attr = lgc.attribute(self.normalized_inp, target=6)
        upsampled_img_attr = LayerAttribution.interpolate(gc_attr, self.normalized_inp.shape[2:])
        (fig, _) = viz.visualize_image_attr_multiple(
            upsampled_img_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
            original_image=self.preproc_img.permute(1, 2, 0).numpy(),
            signs=["all", "positive", "negative"],
            methods=["original_image", "blended_heat_map", "blended_heat_map"],
            titles=["Original Image", "Positive GradCAM ", "Negative GradCAM"],
            use_pyplot=False,
        )
        path = os.path.join(
            os.path.dirname(os.path.abspath(self.output_path)), "Overlay_LayerGradCam.png"
        )
        fig.savefig(path)
        print("[INFO] Overlayed LayerGradCam image saved to path {} : ".format(path))
        return [{self.mapping[str(target)]: str(path)}]

    def explain_handle(self, normalized_inp, raw_data):
        """Captum explanations handler
        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request
        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = self.get_insights(normalized_inp, target=6)
        return output_explain
