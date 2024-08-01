"""
This file is used to train the model.
"""
import os

# Importing the required libraries
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from data_preperation import MNISTDataPreparation
from hyper_parameters import HyperParameters
from tensorboard_logging import TensorboardLogging
from model import NeuralNet

# set up mlflow for tracking
logger = TensorboardLogging(run_name="run-1")

# Prepare Hyperparameters and log to mlflow
hyper_parameters = HyperParameters(batch_size=200)
logger.log_hyper_parameters(hyper_parameters)

# Prepare data
data = MNISTDataPreparation(batch_size=hyper_parameters.batch_size, data_dir='data', is_download=False)
train_loader, test_loader = data.prepare_data()



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and log model summary
model = NeuralNet(hyper_parameters)
model.to(device)
logger.log_model_summary(model, hyper_parameters)
#TODO: solve for this error
# logger.log_model_graph(model)

# Loss, accuracy & optimizer setup
loss_func = nn.CrossEntropyLoss()
accuracy_func = Accuracy(task="multiclass", num_classes=10, average="macro").to(device)
f1_score_func = F1Score(task="multiclass", num_classes=10, average="macro").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hyper_parameters.learning_rate)

# setup variables for print
print_after_n_steps = 100
steps_per_epoch = data.get_steps_per_epoch()

# Training loop
for epoch_idx in range(hyper_parameters.num_epochs):
    # placeholders for accumulated labels, predictions for all batches of training
    batch_train_labels = []
    batch_train_predictions = []
    for batch_idx, (images, labels) in enumerate(train_loader):

        # set the model in training mode
        model.train()

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = loss_func(predictions, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

        # accumulate labels and predictions for all batches of training
        batch_train_labels.append(labels)
        batch_train_predictions.append(predictions)

        # Print the loss every `print_after_n_steps` steps
        if (batch_idx+1) % print_after_n_steps == 0:
            # concatenate all predictions & all labels for all batches of training
            train_predictions = torch.cat(batch_train_predictions, dim=0)
            train_labels = torch.cat(batch_train_labels, dim=0)

            # calculate accumulated loss
            accumulated_loss = loss_func(train_predictions, train_labels)

            # calculate accumulated training accuracy and log to mlflow
            _, predicted_classes = torch.max(train_predictions, dim=1)
            accuracy = accuracy_func(predicted_classes, train_labels)
            f1score = f1_score_func(predicted_classes, train_labels)


            total_step_count = (epoch_idx * steps_per_epoch) + (batch_idx + 1)
            print ('Epoch [{}/{}], Step [{}/{}, total steps = {}], Loss: {:.4f}, f1-score: {:.4f}, accuracy: {:.4f}'
                   .format(epoch_idx+1, hyper_parameters.num_epochs, batch_idx+1, len(train_loader), total_step_count, loss.item(), f1score.item(), accuracy.item()))

            # Log the loss to mlflow
            logger.log_loss("training", accumulated_loss.item(), total_step_count)
            # logger.log_metric("training-accuracy", accuracy.item(), total_step_count)
            # logger.log_metric("training-f1-score", f1score.item(), total_step_count)

    # Validation loop
    # set the model in evaluation mode
    model.eval()
    # placeholders for accumulated labels, predictions for all batches of validation
    all_val_labels = []
    all_val_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            all_val_labels.append(labels)
            all_val_predictions.append(predictions)

        # concatenate all predictions & all labels for all batches of validation
        validation_predictions = torch.cat(all_val_predictions, dim=0)
        validation_labels = torch.cat(all_val_labels, dim=0)


        # calculate validation loss
        validation_loss = loss_func(validation_predictions, validation_labels)

        # calculate validation accuracy
        _, validation_predictions_classes = torch.max(validation_predictions, dim=1)
        validation_accuracy = accuracy_func(validation_predictions_classes, validation_labels)
        validation_f1score = f1_score_func(validation_predictions_classes, validation_labels)

        # Log the accumulated loss and accuracy metrics to mlflow
        logger.log_loss("validation", validation_loss.item(), total_step_count)
        # logger.log_metric("validation-accuracy", validation_accuracy.item(), total_step_count)
        # logger.log_metric("validation-f1-score", validation_f1score.item(), total_step_count)

        print('Validation loss: {:.4f}, f1-score: {:.4f}, accuracy: {:.4f}'
              .format(validation_loss.item(), validation_f1score.item(), validation_accuracy.item()))



logger.stop()