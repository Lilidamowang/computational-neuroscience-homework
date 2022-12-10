from builder.Builder import OperationBuilder, TestBuilder
from product.Operation import Operation
from product.Test import Test
from model.model_rim_okvqa import CMGCNnet_RIM
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
import hues

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
from dataset.okvqa_traindataset import OkvqaTrainDataset
from dataset.okvqa_testdataset import OkvqaTestDataset


class OKVQATestBuilder(TestBuilder):
    __config_path = 'option/config_okvqa.yml'
    __operation = None

    def __init__(self, device, model_path, model, save_path, k=1):
        hues.info("model:", model)
        self.device = device
        self.config = yaml.load(open(self.__config_path))
        self.__operation = Test(device, model_path, model, k)

    def setDataset(self):
        val_dataset = OkvqaTestDataset(self.config, overfit=False, in_memory=True)
        self.__operation.setDataset(val_dataset, "OKVQA")

    def setOption(self):
        option = Option('option/config_okvqa.yml')
        self.__operation.setOption(option)

    def getOperation(self):
        return self.__operation