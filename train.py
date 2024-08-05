"""
This file is used to train the model.
"""
import os
import time

# Importing the required libraries
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from data_preperation import MNISTDataPreparation
from hyper_parameters import HyperParameters
from tensorboard_logging import TensorboardLogging
from model import NeuralNet

# set up mlflow for tracking
logger = TensorboardLogging(run_name="run-4")
logger.setup_gpu_usage_metrics()

# Prepare Hyperparameters and log to mlflow
hyper_parameters = HyperParameters(batch_size=200)

# Prepare data
## If we are using MNIST dataset within MNISTDataPreparation class, argument `data_dir` should be set to "data" and there must be a directory called MNIST that is donloaded.
## If the data is not downloaded, argument `is_download` should be set to True.
data = MNISTDataPreparation(batch_size=hyper_parameters.batch_size, data_dir='data/MNIST_as_CSV', is_download=False)
train_loader, test_loader = data.prepare_data()
classes = data.classes



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = str(device)
hyper_parameters.device = device_name
# Avoid device profiler as it is too heavy. Creates a lot of data, overhead
# logger.setup_device_profiler()
logger.log_hyper_parameters(hyper_parameters)


# Model and log model summary
model = NeuralNet(hyper_parameters)
model.to(device)
logger.log_model_summary(model, hyper_parameters, device_name)
logger.log_model_graph(model, torch.randn(hyper_parameters.batch_size, hyper_parameters.input_size).to(device))

# Loss, accuracy & optimizer setup
loss_func = nn.CrossEntropyLoss()
accuracy_func = Accuracy(task="multiclass", num_classes=10, average="macro").to(device)
f1_score_func = F1Score(task="multiclass", num_classes=10, average="macro").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameters.learning_rate)

# setup variables for print
log_after_n_train_steps = 100
steps_per_epoch = data.get_steps_per_epoch()

# Training loop
for epoch_idx in range(hyper_parameters.num_epochs):
    epoch_start_time = time.time()
    for batch_idx, (batch_train_images, batch_train_labels) in enumerate(train_loader):
        batch_train_start_time = time.time()

        # placeholders for accumulated labels, predictions for batches until logged
        accumulated_batch_train_labels = []
        accumulated_batch_train_predictions = []

        # set the model in training mode
        model.train()

        # Move tensors to the configured device
        batch_train_images = batch_train_images.to(device)
        batch_train_labels = batch_train_labels.to(device)

        # Forward pass
        batch_train_predictions = model(batch_train_images)

        # Calculate loss
        batch_train_loss = loss_func(batch_train_predictions, batch_train_labels)

        # Backward
        optimizer.zero_grad()
        batch_train_loss.backward()

        # Optimize
        optimizer.step()

        # accumulate labels and predictions for all batches of training
        accumulated_batch_train_labels.append(batch_train_labels)
        accumulated_batch_train_predictions.append(batch_train_predictions)

        total_step_count = (epoch_idx * steps_per_epoch) + (batch_idx + 1)

        # Print the loss every `print_after_n_steps` steps
        if (batch_idx+1) % log_after_n_train_steps == 0:
            # concatenate all predictions & all labels for all batches of training
            accumulated_batch_train_predictions = torch.cat(accumulated_batch_train_predictions, dim=0)
            accumulated_batch_train_labels = torch.cat(accumulated_batch_train_labels, dim=0)

            # calculate accumulated loss
            accumulated_batch_train_loss = loss_func(accumulated_batch_train_predictions, accumulated_batch_train_labels)

            # calculate accumulated training accuracy and log to mlflow
            _, accumulated_batch_train_predicted_classes_indices = torch.max(accumulated_batch_train_predictions, dim=1)
            accumulated_batch_train_accuracy = accuracy_func(accumulated_batch_train_predicted_classes_indices, accumulated_batch_train_labels)
            accumulated_batch_train_f1score = f1_score_func(accumulated_batch_train_predicted_classes_indices, accumulated_batch_train_labels)



            print ("Epoch [{}/{}], Step [{}/{}, total steps = {}], Accumulated batches' Loss: {:.4f}, Accumulated batches' f1-score: {:.4f}, Accumulated batches' accuracy: {:.4f}"
                   .format(epoch_idx+1, hyper_parameters.num_epochs, batch_idx+1, len(train_loader), total_step_count, accumulated_batch_train_loss.item(), accumulated_batch_train_f1score.item(), accumulated_batch_train_accuracy.item()))

            # Log the loss to mlflow
            logger.log_loss("Accumulated training batches' loss", accumulated_batch_train_loss.item(), total_step_count)
            # logger.log_metric("training-accuracy", accuracy.item(), total_step_count)
            # logger.log_metric("training-f1-score", f1score.item(), total_step_count)
            # Avoid device profiler as it is too heavy. Creates a lot of data, overhead
            # logger.device_profiler_step()

        batch_train_end_time = time.time()
        logger.log_execution_time("Training batch", int(batch_train_end_time - batch_train_start_time),
                                  total_step_count)

    # Validation loop
    validation_start_time = time.time()
    # set the model in evaluation mode
    model.eval()
    # placeholders for accumulated labels, predictions for all batches of validation
    validation_labels = []
    validation_predictions = []
    with torch.no_grad():
        for batch_validation_images, batch_validation_labels in test_loader:    # validation_data = test_data in this case
            batch_validation_images = batch_validation_images.to(device)
            batch_validation_labels = batch_validation_labels.to(device)
            batch_validation_predictions = model(batch_validation_images)
            validation_labels.append(batch_validation_labels)
            validation_predictions.append(batch_validation_predictions)

        # concatenate all predictions & all labels for all batches of validation
        validation_predictions = torch.cat(validation_predictions, dim=0)
        validation_labels = torch.cat(validation_labels, dim=0)


        # calculate validation loss
        validation_loss = loss_func(validation_predictions, validation_labels)

        # calculate validation accuracy
        _, validation_predictions_classes_indices = torch.max(validation_predictions, dim=1)
        validation_accuracy = accuracy_func(validation_predictions_classes_indices, validation_labels)
        validation_f1score = f1_score_func(validation_predictions_classes_indices, validation_labels)



        print('Validation loss: {:.4f}, f1-score: {:.4f}, accuracy: {:.4f}'
              .format(validation_loss.item(), validation_f1score.item(), validation_accuracy.item()))
        # Log the accumulated loss and accuracy metrics to mlflow
        logger.log_loss("validation loss", validation_loss.item(), total_step_count)
        # logger.log_metric("validation-accuracy", validation_accuracy.item(), total_step_count)
        # logger.log_metric("validation-f1-score", validation_f1score.item(), total_step_count)

        logger.visualize_embeddings(validation_predictions, validation_labels, total_step_count, classes=classes)
        # Avoid device profiler as it is too heavy. Creates a lot of data, overhead
        # logger.device_profiler_step()
        # logger.device_profiler_step()

    validation_end_time = time.time()
    logger.log_execution_time("Full Validation", int(validation_end_time - validation_start_time),
                              total_step_count)

    epoch_end_time = time.time()
    print("Time taken for epoch: ", epoch_end_time - epoch_start_time)
    logger.log_execution_time("Full Epoch", int(epoch_end_time - epoch_start_time),
                              total_step_count)


# Plot PR curves for all classes on test data. This will be logged to Tensorboard.
# Use validation data if test data is not available.
# convert predictions to probabilities using softmax (each class has a probability and value in [0,1])
validation_probabilities = torch.nn.functional.softmax(validation_predictions, dim=1)
logger.add_pr_curves(validation_probabilities, validation_labels, total_step_count, hyper_parameters.num_classes)
logger.stop()