import os
import json
from pprint import pprint
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm

from coral_reef.constants import paths, mapping
from coral_reef.constants import strings as STR

from coral_reef.visualisation import visualisation

from coral_reef.ml.data_set import DictArrayDataSet, RandomCrop, Resize, custom_collate, ToTensor, Flip, Normalize
from coral_reef.ml.utils import load_state_dict, Saver, calculate_class_weights


sys.path.extend([paths.DEEPLAB_FOLDER_PATH, os.path.join(paths.DEEPLAB_FOLDER_PATH, "utils")])

from modeling.deeplab import DeepLab
from loss import SegmentationLosses
from lr_scheduler import LR_Scheduler
from metrics import Evaluator



class Trainer:

    def __init__(self, data_train, data_valid, image_base_dir, instructions):
        """

        :param data_train:
        :param data_valid:
        :param image_base_dir: A directory with all CLEF images (for training and validating)
        :param instructions (dict): A dictionary containing instructions to train. It has
        the following keys:
            epochs
            model_name
            nn_input_shape
            state_dict_file_path (default = None)
            crops
            images_per_batch
            batch_size
            backbone (default = resnet)
            deeplab_output_stride (default = 16)
            learning_rate (default = 1e-05)
            multi_gpu (default = False)
            class_stats_file_path
            use_lr_scheduler (default = True)
        """

        self.image_base_dir = image_base_dir
        self.data_valid = data_valid
        self.instructions = instructions

        # specify model save dir
        self.model_name = instructions[STR.MODEL_NAME]
        # now = time.localtime()
        # start_time = "{}-{}-{}T{}:{}:{}".format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min,
        #                                         now.tm_sec)
        experiment_folder_path = os.path.join(paths.MODELS_FOLDER_PATH, self.model_name)

        if os.path.exists(experiment_folder_path):
            Warning("Experiment folder exists already. Files might be overwritten")
        os.makedirs(experiment_folder_path, exist_ok=True)

        # define saver and save instructions
        self.saver = Saver(folder_path=experiment_folder_path,
                           instructions=instructions)
        self.saver.save_instructions()

        # define Tensorboard Summary
        self.writer = SummaryWriter(log_dir=experiment_folder_path)

        nn_input_size = instructions[STR.NN_INPUT_SIZE]
        state_dict_file_path = instructions.get(STR.STATE_DICT_FILE_PATH, None)

        self.colour_mapping = mapping.get_colour_mapping()

        # define transformers for training
        crops_per_image = instructions.get(STR.CROPS_PER_IMAGE, 10)

        apply_random_cropping = (STR.CROPS_PER_IMAGE in instructions.keys()) and \
                                (STR.IMAGES_PER_BATCH in instructions.keys())

        print("{}applying random cropping".format("" if apply_random_cropping else "_NOT_ "))

        t = [Normalize()]
        if apply_random_cropping:
            t.append(RandomCrop(
                min_size=instructions.get(STR.CROP_SIZE_MIN, 400),
                max_size=instructions.get(STR.CROP_SIZE_MAX, 1000),
                crop_count=crops_per_image))
        t += [Resize(nn_input_size),
              Flip(p_vertical=0.2, p_horizontal=0.5),
              ToTensor()]

        transformations_train = transforms.Compose(t)

        # define transformers for validation
        transformations_valid = transforms.Compose([Normalize(), Resize(nn_input_size), ToTensor()])

        # set up data loaders
        dataset_train = DictArrayDataSet(image_base_dir=image_base_dir,
                                         data=data_train,
                                         num_classes=len(self.colour_mapping.keys()),
                                         transformation=transformations_train)

        # define batch sizes
        self.batch_size = instructions[STR.BATCH_SIZE]

        if apply_random_cropping:
            self.data_loader_train = DataLoader(dataset=dataset_train,
                                                batch_size=instructions[STR.IMAGES_PER_BATCH],
                                                shuffle=True,
                                                collate_fn=custom_collate)
        else:
            self.data_loader_train = DataLoader(dataset=dataset_train,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                collate_fn=custom_collate)

        dataset_valid = DictArrayDataSet(image_base_dir=image_base_dir,
                                         data=data_valid,
                                         num_classes=len(self.colour_mapping.keys()),
                                         transformation=transformations_valid)

        self.data_loader_valid = DataLoader(dataset=dataset_valid,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            collate_fn=custom_collate)

        self.num_classes = dataset_train.num_classes()

        # define model
        print("Building model")
        self.model = DeepLab(num_classes=self.num_classes,
                             backbone=instructions.get(STR.BACKBONE, "resnet"),
                             output_stride=instructions.get(STR.DEEPLAB_OUTPUT_STRIDE, 16))

        # load weights
        if state_dict_file_path is not None:
            print("loading state_dict from:")
            print(state_dict_file_path)
            load_state_dict(self.model, state_dict_file_path)

        learning_rate = instructions.get(STR.LEARNING_RATE, 1e-5)
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': learning_rate},
                        {'params': self.model.get_10x_lr_params(), 'lr': learning_rate}]

        # choose gpu or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if instructions.get(STR.MULTI_GPU, False):
            if torch.cuda.device_count() > 1:
                print("Using ", torch.cuda.device_count(), " GPUs!")
                self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        # Define Optimizer
        self.optimizer = torch.optim.SGD(train_params,
                                         momentum=0.9,
                                         weight_decay=5e-4,
                                         nesterov=False)

        # calculate class weights
        if instructions.get(STR.CLASS_STATS_FILE_PATH, None):

            class_weights = calculate_class_weights(instructions[STR.CLASS_STATS_FILE_PATH],
                                                    self.colour_mapping,
                                                    modifier=instructions.get(STR.LOSS_WEIGHT_MODIFIER, 1.01))

            class_weights = torch.from_numpy(class_weights.astype(np.float32))
        else:
            class_weights = None
        self.criterion = SegmentationLosses(weight=class_weights, cuda=self.device.type != "cpu").build_loss()

        # Define Evaluator
        self.evaluator = Evaluator(self.num_classes)

        # Define lr scheduler
        self.scheduler = None
        if instructions.get(STR.USE_LR_SCHEDULER, True):
            self.scheduler = LR_Scheduler(mode="cos",
                                          base_lr=learning_rate,
                                          num_epochs=instructions[STR.EPOCHS],
                                          iters_per_epoch=len(self.data_loader_train))

        # print information before training start
        print("-" * 60)
        print("instructions")
        pprint(instructions)
        model_parameters = sum([p.nelement() for p in self.model.parameters()])
        print("Model parameters: {:.2E}".format(model_parameters))

        self.best_prediction = 0.0

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0

        # create a progress bar
        pbar = tqdm(self.data_loader_train)
        num_batches_train = len(self.data_loader_train)

        # go through each item in the training data
        for i, sample in enumerate(pbar):
            # set input and target
            nn_input = sample[STR.NN_INPUT].to(self.device)
            nn_target = sample[STR.NN_TARGET].to(self.device, dtype=torch.float)

            if self.scheduler:
                self.scheduler(self.optimizer, i, epoch, self.best_prediction)

            # run model
            output = self.model(nn_input)

            # calc losses
            loss = self.criterion(output, nn_target)
            # # save step losses
            # combined_loss_steps.append(float(loss))
            # regression_loss_steps.append(float(regression_loss))
            # classification_loss_steps.append(float(classification_loss))

            train_loss += loss.item()
            pbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_batches_train * epoch)

            # calculate gradient and update model weights
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print("[Epoch: {}, num images/crops: {}]".format(epoch, num_batches_train * self.batch_size))

        print("Loss: {:.2f}".format(train_loss))

    def validation(self, epoch):

        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0

        pbar = tqdm(self.data_loader_valid, desc='\r')
        num_batches_val = len(self.data_loader_valid)

        for i, sample in enumerate(pbar):
            # set input and target
            nn_input = sample[STR.NN_INPUT].to(self.device)
            nn_target = sample[STR.NN_TARGET].to(self.device, dtype=torch.float)

            with torch.no_grad():
                output = self.model(nn_input)

            loss = self.criterion(output, nn_target)
            test_loss += loss.item()
            pbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            nn_target = nn_target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(nn_target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print("[Epoch: {}, num crops: {}]".format(epoch, num_batches_val * self.batch_size))
        print("Acc:{:.2f}, Acc_class:{:.2f}, mIoU:{:.2f}, fwIoU: {:.2f}".format(Acc, Acc_class, mIoU, FWIoU))
        print("Loss: {:.2f}".format(test_loss))

        new_pred = mIoU
        is_best = new_pred > self.best_prediction
        if is_best:
            self.best_prediction = new_pred
        self.saver.save_checkpoint(self.model, is_best, epoch)


def train(data_train, data_valid, image_base_dir, instructions):
    trainer = Trainer(data_train, data_valid, image_base_dir, instructions)

    epochs = instructions[STR.EPOCHS]
    for epoch in range(1, epochs + 1):
        trainer.train(epoch)
        trainer.validation(epoch)
