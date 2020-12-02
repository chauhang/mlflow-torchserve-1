# Model drift Analysis

Machine learning lifecycle starts with data collection and processing and ends with the deployment of the tested model.
The performance of every machine learning model will deteriorate with time. Some may decline gradually and some may decline faster. 
The main reason is because the relationship between the target variable and the independent variable may not remain the same all the time. 
One of the main reason for such drifts in model performance happens because When the statistical properties of the independent variables change with time. 
(Example: changing of patterns of data due to seasonality).In this example, we are going to perform a model drift analysis with PyTorch model. 
The approach here that we have used to address the drift issue is to continuously re-train the model.An estimate could be drawn as to when the deterioration could occur. 
Based on this the model is pro-actively re-trained as to eliminate the risks of the drift.The model is then tested in order of the blocks and for each block the accuracy is recorded.
The drift is identified as soon as any of the test accuracy of the blocks are below the threshold accuracy.We have used the "test_accuracy" as the KPI on which the model performance is evaluated. 
When the accuracy falls below a threshold level the model is re-trained.For every step in which the drifts are identified this process is repeated till acceptable performance is achieved.

This data was collected from the Australian New South Wales Electricity Market. In this market, prices are not fixed and are affected by demand and supply of the market.
They are set every five minutes. The ELEC dataset contains 45, 312 instances.The class label identifies the change of the price relative to a moving average of the last 24 hours(namely UP and Down). 
You can download the dataset [here](https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/elecNormNew.arff.zip/download/).
To select the best performing model we have used the Ax hyperparameter optimization, which will help us detect the best set of parameters to bring the best performance.
The autolog function is incorporated to log all the standard parameters, metrics and artifacts in MLflow, associated with each run of the model. 


Major Steps:

1. Training the Base Model

2. Deploying the model in TorchServe and evaluating the model.

3. Retraining the model if the model performance is below the threhold performance level.



### Running the code

## Setting up Environment Variable

Set up the experiment name environment variable as follows:

`export MLFLOW_EXPERIMENT_NAME=drift1`

To run the example via MLflow, navigate to the `examples/ModelDriftAnalysis/` directory and run the command

```
mlflow run .

```

This will run `base_model.py` with the default set of parameters such as `--max_epochs=3`. You can see the default value in the MLproject file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X -P mlflow_experiment_name=Y
```

where X is your desired value for max_epochs and Y is the name of the experiment.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

You can also run the exmaple directly with custom parameters:

```
python base_model.py \
    --max_epochs 3  \
    --mlflow_experiment_name drift1
```

## Deploying the model in TorchServe 

Run the following command to deploy the model in TorchServe.

```
python deploy_model.py \
    --max_epochs 3  \
    --test_data_rows 100 \
    --block_size 20 \
    --threshold_accuracy .55 \
    --register_model_name drift_model \
    --mlflow_experiment_name drift
```

At the end of this, it will prompt you whether you need to retrain the model based on the performance and the threshold accuracy.

## Retrain the Model

If you need to retrain the model, run the following 

`python retrain_model.py --max_epochs 3  --mlflow_experiment_name drift2 --total_trials 2`

This will retrain the model and expected to improve the performance of the model. 

In the MLflow UI, the Base model and the retrained model appears as Parent and Chil runs as shown below:

![drift1](https://user-images.githubusercontent.com/51693147/100898152-78476780-34e6-11eb-8f7f-dea385e00fad.JPG)

If you compare the child runs and the parent run in the UI you get :

![drift2](https://user-images.githubusercontent.com/51693147/100898199-81d0cf80-34e6-11eb-8822-2ace68c44985.JPG)


For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.