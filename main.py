from director.Director import Director
from builder.FVQATrainOperationBuilder import FVQATrainOperationBuilder
from builder.OKVQATrainOperationBuilder import OKVQATrainOperationBuilder
from builder.OKVQATestBuilder import OKVQATestBuilder
from option.Args import Args

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

'''
model_load_path="/data2/yjgroup/lyl/result/RIM_PR_model/OKVQA/new_Code/checkpoint_21.pth"
'''


def OKVQATrain(args=None):
    hues.info(f'Training in {args.getDevice()}')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device(args.getDevice())
    weight = torch.FloatTensor([0.9, 0.1]).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weight)

    # 构建 OKVQA
    director = Director()
    builder = OKVQATrainOperationBuilder(device, loss_function, 0, 100, model_type=args.getModelType(), model=args.getModel()
                                            , model_load_path=args.getModelLoadPath(), save_path=args.getSavePath())
    director.makeOperation(builder)
    operation_okvqa = builder.getOperation()

    # 训练
    operation_okvqa.train()


def OKVQATest(args=None):
    hues.info('Testing')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    device = torch.device(args.getDevice())

    director = Director()
    builder = OKVQATestBuilder(device, args.getModelLoadPath(), args.getModel(), save_path=args.getSavePath())
    director.makeTest(builder)
    operation = builder.getOperation()

    operation.startTest()

if __name__ == '__main__':
    args = Args()
    if args.getTrain() == 1:
        OKVQATrain(args)
    else:
        OKVQATest(args)
