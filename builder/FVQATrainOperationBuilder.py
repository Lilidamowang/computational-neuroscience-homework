from builder.Builder import OperationBuilder
from product.Operation import Operation
from model.model_rim_fvqa import CMGCNnet_RIM_FVQA
from option.Option import Option

import os
import json
import yaml
import argparse
import hues
import numpy as np

from math import log
import dgl
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint
from util.myFun import FunClo
from dataset.fvqa_traindataset import FvqaTrainDataset
from dataset.fvqa_testdataset import FvqaTestDataset


class FVQATrainOperationBuilder(OperationBuilder):
    __config_path = 'option/config_fvqa_lyl.yml'
    __operation = None

    def __init__(self, device, loss_function, start_epoch, num_epoch):
        self.device = device
        self.config = yaml.load(open(self.__config_path))
        self.__operation = Operation(device, loss_function, start_epoch, num_epoch)

    def setModel(self):

        glovevocabulary = Vocabulary(self.config["dataset"]["word_counts_json"],
                                     min_count=self.config["dataset"]["vocab_min_count"])
        glove = np.load(self.config['dataset']['glove_vec_path'])
        glove = torch.Tensor(glove)
        model = CMGCNnet_RIM_FVQA(self.config,
                                  que_vocabulary=glovevocabulary,
                                  glove=glove,
                                  device=self.device,
                                  n_layers=5,
                                  num_units=6,
                                  hidden_size=128)

        optimizer = optim.Adamax(model.parameters(),
                                 lr=0.001)
        self.__operation.setModel(model, optimizer)

    def setOption(self):
        option = Option('option/config_fvqa_lyl.yml')
        self.__operation.setOption(option)

    def setTrainDataset(self):
        train_dataset = FvqaTrainDataset(self.config, overfit=False)
        self.__operation.setTrainDataset(train_dataset)

    def setTestDataset(self):
        test_dataset = FvqaTestDataset(self.config, overfit=False)
        self.__operation.setTestDataset(test_dataset)

    def setSavePath(self):
        self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/FVQA/new_Code/')

    def getOperation(self):
        return self.__operation
