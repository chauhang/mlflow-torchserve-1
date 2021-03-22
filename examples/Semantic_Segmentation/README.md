#Semantic Segmentation with Captum and MLflow

In this example, we demonstrate applying Captum to semantic segmentation task, to understand what pixels and regions contribute to the labeling of a particular class. We explore applying GradCAM as well as Feature Ablation to a pretrained Fully-Convolutional Network model with a ResNet-101 backbone. You can find more details [here][https://captum.ai/tutorials/Segmentation_Interpret]


### Running the code

To run the example via MLflow, navigate to the `examples/Semantic_Segmentation/` directory and run the command

```
mlflow run .

```

This will run `segmentation_model.py` with the default set of parameters such as `--target=6`. You can see the default value in the MLproject file. where target can be any of the trained class out of 21 in the model.

In order to run the file with custom parameters, run the command

```
mlflow run . -P img_url="url" -P target=8
```

where url can be any test image url .

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

## Starting TorchServe

Download the pre-trained fcn_resnet_101_coco image segmentation model's state_dict from the following URL:
https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
`wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth`

## Creating a new deployment

Run the following command to create a new deployment named `segmentation`

The `index_to_name.json` file is the mapping file, which will convert the discrete output of the model to one of the flower species
based on the predefined mapping.

`torch-model-archiver --model-name segmentation --version 1.0 --model-file model.py --serialized-file fcn_resnet101_coco-7ecb50ca.pth --handler segmentation_handler.py --extra-files fcn.py,intermediate_layer_getter.py,index_to_name
mkdir model_store
mv fcn_resnet_101.mar model_store/
torchserve --start --model-store model_store --ts-config config.properties"`

## Running explanations based on deployed model
	curl http://127.0.0.1:8080/explanations/segmentation -T data/input_1.json


Run the following command to invoke prediction of our sample input, whose output is stored in output.json file.

`mlflow deployments predict --name iris_test --target torchserve --input-path sample.json  --output-path output.json`

The model will classify the flower species based on the input test data as one among the three types and store it in `output.json`