# Multi-Layer Perceptron model for multi-class classification

This is a multi-layer perceptron model for multi-class classification using the pytorch library.

## Dataset
MNIST: Handwritten digits dataset

## Logging and Monitoring
We use MLflow for logging and monitoring the model training process.

## TODO
- Add Tensorboard support
- Add support for custom datasets
- Add logging dataset summary
- Add logging system metrics
- Create a separate class for accuracy metrics
- Loss function and Optimizer should be configurable and added to logs1
- Add logging device information
- Add logging execution time
- This repository is supposed to take 1-D data as input. But we are inputting images(MNIST) and flattening them. We should flatten the image while creating the dataset itself. Thus making the code available for any custom 1-D data.
- Add test, inference & batch inference scripts