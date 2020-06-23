# -*- coding: utf-8 -*-
# Radhika Mattoo, radhika095@gmail.com
# Code taken from:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py

import time
import os
import copy
import torch
import utils
from utils import VisdomLinePlotter, get_data_transforms, imshow, load_model
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            class_to_idx = classes_to_idx[phase]
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            step = 0
            total = len(dataloaders[phase].dataset)
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                step += 1
                if step == total:
                    break
            print()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Keep track of loss per epoch
            if phase == 'val':
                plotter.plot('loss', 'val', 'Class Loss', epoch, epoch_loss)
                plotter.plot('acc', 'val', 'Class Accuracy', epoch, epoch_acc.cpu().detach().numpy())
                name_preds = [class_to_idx[p] for p in preds.detach().cpu().numpy()]
                labels = [class_to_idx[l] for l in labels.detach().cpu().numpy()]
                caption = ""
                for idx in range(len(labels)):
                    label = labels[idx]
                    prediction = name_preds[idx]
                    caption += label + " vs. " + prediction + " |\t"
                plotter.images(inputs, win='x', opts=dict(title='Label vs. Prediction in epoch {}'.format(epoch), caption=caption))

            else:
                plotter.plot('loss', 'train', 'Class Loss', epoch, epoch_loss)
        print( '\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    actuals, probabilities = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            sm = torch.nn.Softmax()
            probabilities = sm(outputs)
             #Converted to probabilities

            for j in range(inputs.size()[0]):
                images_so_far += 1
                probability = probabilities[j][preds[j]]
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, {}%'.format(class_names[preds[j]], probability))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # EXECUTION
    parser.add_argument(
        '--mode',
        nargs='?',
        choices=['train', 'eval'],
        help='Execution mode')
    # DATA INFO
    parser.add_argument(
        '--data-dir',
        default='data',
        type=str,
        help='Directory where train/val/test data is stored')

    # TRAIN HYPERPARAMS
    parser.add_argument(
        '--num-epochs',
        default=10,
        type=int,
        help='Number of epochs')
    parser.add_argument(
        '--learning-rate',
        default=0.00001,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum')
    parser.add_argument(
        '--decay',
        default=False,
        type=bool,
        help='Whether the learning rate should decay during training or not')
    parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='Learning rate decay factor')
    parser.add_argument(
        '--step-size',
        default=7,
        type=int,
        help='How often (in epochs) to decay the learning rate')
    parser.add_argument(
        '--batch-size',
        default=32,
        type=int,
        help='Number of images trained on before backprop is applied')

    # EVAL OPTIONS
    parser.add_argument(
        '--model-file',
        default='',
        type=str,
        help='Absolute path to model file to use for evaluation')
    parser.add_argument(
        '--num-images',
        default=10,
        type=int,
        help='Number of images to use for evaluation')
    args = parser.parse_args()

    # CPU or GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device {}".format(device))

    # Args for loading data
    data_dir = args.data_dir
    batch_size = args.batch_size
    data_transforms = get_data_transforms()

    # Load in and transform data
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    # Maps class_idx to its string label
    classes_to_idx = {x: { class_idx: class_name for class_name, class_idx in image_datasets[x].class_to_idx .items()} for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    if args.mode == "train":
        # Visdom for plots
        global plotter
        plotter = utils.VisdomLinePlotter(env_name='NameThatDog Plots')

        # Extract args
        num_epochs = args.num_epochs
        learning_rate = args.learning_rate
        momentum = args.momentum
        decay = args.decay
        gamma = args.gamma
        step_size = args.step_size

        print("Training new final layer of Resnet18 model for {} Epochs with hyperparams: LR: {}, Momentum: {}, Decay: {}, Gamma: {}, Step Size: {}, Batch Size: {}".format(num_epochs, learning_rate, momentum, decay, gamma, step_size, batch_size))

        # Download pretrained resnet18 model
        model_conv = torchvision.models.resnet18(pretrained=True)

        # Freeze existing layers
        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features

        # Create new final layer to be trained/optimized on
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))

        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opoosed to before.
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum)

        exp_lr_scheduler = None
        if decay:
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

        # Train model
        model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=num_epochs)

        # Save
        timestamp = str(int(time.time()))
        today = datetime.today().strftime('%Y-%m-%d')
        model_filename = os.path.join('models', 'namethatdog_' + today + '_' + timestamp + '.pt')
        print('Saving model to {}'.format(model_filename))

        torch.save(model_conv, model_filename)
    elif args.mode == "eval":
        path_to_model = args.model_file
        num_images = args.num_images
        print("Evaluating model file {} with {} images".format(path_to_model, num_images))
        try:
            model_conv = load_model_for_evaluation(path_to_model)
        except Exception as e:
            print("Error loading model: {}".format(e))
            exit()
        evaluate_model(model_conv, num_images)
        plt.ioff()
        plt.show()
    else:
        print("No valid mode selected, exiting")
        exit()
