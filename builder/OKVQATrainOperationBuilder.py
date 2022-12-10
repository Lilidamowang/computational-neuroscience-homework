from builder.Builder import OperationBuilder
from product.Operation import Operation
from model.model_rim_okvqa import CMGCNnet_RIM
from model.model_baseline import CMGCNnet
from model.model_test_1 import MMR_Test
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
from dataset.okvqa_traindataset import OkvqaTrainDataset
from dataset.okvqa_testdataset import OkvqaTestDataset


class OKVQATrainOperationBuilder(OperationBuilder):
    __config_path = 'option/config_okvqa.yml'
    __model_load_path = ""
    __operation = None
    __model = None

    # model : [base, MMR]
    def __init__(self, device, loss_function, start_epoch, num_epoch, model_type, model, save_path,model_load_path=""):
        hues.info('loading model', model_type)
        self.device = device
        self.model_type = model_type
        self.config = yaml.load(open(self.__config_path))
        self.__operation = Operation(device, loss_function, start_epoch, num_epoch)
        self.__model_load_path = model_load_path
        self.model = model
        self.save_path = save_path

    def setModel(self):
        glovevocabulary = Vocabulary(self.config["dataset"]["word_counts_json"],
                                     min_count=self.config["dataset"]["vocab_min_count"])
        glove = np.load(self.config['dataset']['glove_vec_path'])
        glove = torch.Tensor(glove)
        if self.model == 'MMR':
            hues.info('loading MMR model')
            model = CMGCNnet_RIM(self.config,
                                 que_vocabulary=glovevocabulary,
                                 glove=glove,
                                 device=self.device,
                                 model_type=self.model_type)
        elif self.model == 'base':
            hues.info('loading base model')
            model = CMGCNnet(self.config,
                             que_vocabulary=glovevocabulary,
                             glove=glove,
                             device=self.device)
        elif self.model == 'Test1':
            hues.info('loading Test1 model')
            model = MMR_Test(self.config,
                                 que_vocabulary=glovevocabulary,
                                 glove=glove,
                                 device=self.device,
                                 model_type=self.model_type)
        start_epoch = 0
        if self.__model_load_path != "":
            start_epoch = int(self.__model_load_path.split("_")[-1][:-4]) + 1
            model_state_dict, optimizer_state_dict = load_checkpoint(self.__model_load_path)
            model.load_state_dict(model_state_dict)
            model = model.to(self.device)
            optimizer = optim.Adamax(model.parameters(),
                                     lr=0.001)
            optimizer.load_state_dict(optimizer_state_dict)
        else:
            model = model.to(self.device)
            # @TODO 改变了lr 0.001 -> 0.01
            optimizer = optim.Adamax(model.parameters(),
                                     lr=0.001)
        self.__operation.setModel(model, optimizer, start_epoch)

    def setOption(self):
        option = Option('option/config_okvqa.yml')
        self.__operation.setOption(option)

    def setTrainDataset(self):
        train_dataset = OkvqaTrainDataset(self.config, overfit=False)
        self.__operation.setTrainDataset(train_dataset)

    def setTestDataset(self):
        test_dataset = OkvqaTestDataset(self.config, overfit=False)
        self.__operation.setTestDataset(test_dataset)

    def setSavePath(self):
        if(self.save_path == ''):
            if self.model == 'MMR':
                if self.model_type == 1:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code/')
                elif self.model_type == 2:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code_3_2/')
                elif self.model_type == 3:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code_5_3/')
                elif self.model_type == 4:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code_6_2/')
                elif self.model_type == 5:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code_attention_8_4_new_action3/')
                elif self.model_type == 6:
                    self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/model_type6_in_action5/')
            elif self.model == 'base':
                hues.info('model saved in : PR_model/OKVQA/base/')
                self.__operation.setSavePath('/home/data/yjgroup/lyl/result/PR_model/OKVQA/base/')
            elif self.model == 'Test1':
                hues.info('model saved in /home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/test1/')
                self.__operation.setSavePath('/home/data/yjgroup/lyl/result/RIM_PR_model/OKVQA/test1/')
        else:
            hues.info(f"model saved in : {self.save_path}")
            self.__operation.setSavePath(self.save_path)

    def getOperation(self):
        return self.__operation
