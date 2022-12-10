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
from scipy.stats import norm

from bisect import bisect
from util.vocabulary import Vocabulary
from util.checkpointing import CheckpointManager, load_checkpoint

def get_gaussions(batch, k):
    res = []
    mu = 0
    sigma = 20
    for i in range(k):
        res.append(norm.pdf(i, loc=mu, scale=sigma))
    res_tensor = torch.tensor(res)
    res_tensor = res_tensor.unsqueeze(dim=0).repeat(batch, 1) # shape = [11, 100]
    res_tensor_softmax = torch.softmax(res_tensor, dim=1)
    return res_tensor_softmax

def collate_fn(batch):  # 二次处理batch数据，转换为需要的形式
    res = {}
    qid_list = []
    question_list = []
    question_length_list = []
    img_features_list = []
    img_relations_list = []

    fact_num_nodes_list = []
    facts_node_features_list = []
    facts_e1ids_list = []
    facts_e2ids_list = []
    facts_answer_list = []
    facts_answer_id_list = []

    semantic_num_nodes_list = []
    semantic_node_features_list = []
    semantic_e1ids_list = []
    semantic_e2ids_list = []
    semantic_edge_features_list = []
    semantic_num_nodes_list = []

    for item in batch:
        # question
        qid = item['id']
        qid_list.append(qid)

        question = item['question']
        question_list.append(question)

        question_length = item['question_length']
        question_length_list.append(question_length)

        # image
        img_features = item['img_features']
        img_features_list.append(img_features)

        img_relations = item['img_relations']
        img_relations_list.append(img_relations)

        # fact
        fact_num_nodes = item['facts_num_nodes']
        fact_num_nodes_list.append(fact_num_nodes)

        facts_node_features = item['facts_node_features']
        facts_node_features_list.append(facts_node_features)

        facts_e1ids = item['facts_e1ids']
        facts_e1ids_list.append(facts_e1ids)

        facts_e2ids = item['facts_e2ids']
        facts_e2ids_list.append(facts_e2ids)

        facts_answer = item['facts_answer']
        facts_answer_list.append(facts_answer)

        facts_answer_id = item['facts_answer_id']
        facts_answer_id_list.append(facts_answer_id)

        # semantic
        semantic_num_nodes = item['semantic_num_nodes']
        semantic_num_nodes_list.append(semantic_num_nodes)

        semantic_node_features = item['semantic_node_features']
        semantic_node_features_list.append(semantic_node_features)

        semantic_e1ids = item['semantic_e1ids']
        semantic_e1ids_list.append(semantic_e1ids)

        semantic_e2ids = item['semantic_e2ids']
        semantic_e2ids_list.append(semantic_e2ids)

        semantic_edge_features = item['semantic_edge_features']
        semantic_edge_features_list.append(semantic_edge_features)

    res['id_list'] = qid_list
    res['question_list'] = question_list
    res['question_length_list'] = question_length_list
    res['features_list'] = img_features_list
    res['img_relations_list'] = img_relations_list
    res['facts_num_nodes_list'] = fact_num_nodes_list
    res['facts_node_features_list'] = facts_node_features_list
    res['facts_e1ids_list'] = facts_e1ids_list
    res['facts_e2ids_list'] = facts_e2ids_list
    res['facts_answer_list'] = facts_answer_list
    res['facts_answer_id_list'] = facts_answer_id_list
    res['semantic_node_features_list'] = semantic_node_features_list
    res['semantic_e1ids_list'] = semantic_e1ids_list
    res['semantic_e2ids_list'] = semantic_e2ids_list
    res['semantic_edge_features_list'] = semantic_edge_features_list
    res['semantic_num_nodes_list'] = semantic_num_nodes_list
    return res


class Operation:
    # 共享参数
    __device = None
    __optimizer = None
    __loss_function = None
    __scheduler = None
    __start_epoch = None
    __num_epoch = None

    # 特定参数
    __model = None
    __option = None
    __train_dataset = None
    __test_dataset = None
    __save_path = None

    iterations = None

    def __init__(self, device, loss_function, start_epoch, num_epoch):
        self.__device = device
        self.__loss_function = loss_function
        self.__start_epoch = start_epoch
        self.__num_epoch = num_epoch

    def setModel(self, model, optimizer, start_epoch):
        self.__model = model
        self.__optimizer = optimizer
        self.__start_epoch = start_epoch

    def setOption(self, option):
        self.__option = option

    def setTrainDataset(self, train_dataset):
        self.__train_dataset = train_dataset

    def setTestDataset(self, test_dataset):
        self.__test_dataset = test_dataset

    def setSavePath(self, save_path):
        self.__save_path = save_path

    # 计算batch的loss
    @staticmethod
    def cal_batch_loss(fact_batch_graph, batch, device, loss_fn, ques_state_weight_softmax, loss_fn_weight = nn.KLDivLoss(reduction='sum')):
        answers = batch['facts_answer_list']
        # hues.info(len(answers));input('s')  11
        # hues.info(answers[0].shape);input('s')  size = 56

        fact_graphs = dgl.unbatch(fact_batch_graph)
        batch_loss = torch.tensor(0).to(device)
        #weight_loss = torch.tensor(0).to(device)
        #alpha = 0.85
        #target_weights_softmax = get_gaussions(batch = ques_state_weight_softmax.shape[0], k=ques_state_weight_softmax.shape[1]).float().to(device)
        #ques_state_weight_softmax = ques_state_weight_softmax.squeeze(dim=-1).log().float()  # shape = [11, 100]

        for i, fact_graph in enumerate(fact_graphs):
            ''' vqa答案预测部分 '''
            pred = fact_graph.ndata['h']  # (n,2)  当前item有n个候选答案，每个候选答案是与不是答案的概率
            answer = answers[i].long().to(device)  # [n] n个候选答案，是答案的为1，不是答案的为0. 既下标为1的表示正确答案
            loss = loss_fn(pred, answer)  # batch中 一个问题的loss
            batch_loss = batch_loss + loss
        
        ''' weight_loss部分 '''
        #weight_loss = loss_fn_weight(ques_state_weight_softmax, target_weights_softmax)

        batch_loss_avg = (batch_loss / len(answers))
        #weight_loss_avg = (weight_loss / 1)
        #total_loss = (alpha*batch_loss_avg) + ((1-alpha)*weight_loss_avg)
        return batch_loss_avg  # 返回一个batch的avg_loss

    # 计算正确率
    @staticmethod
    def cal_acc(answers, preds):
        all_num = len(preds)
        acc_num_1 = 0
        for i, answer_id in enumerate(answers):
            pred = preds[i]  # (num_nodes)  找到i个问题的所有候选答案
            try:
                # top@1
                _, idx_1 = torch.topk(pred, k=1)  # 找到得分最高的候选答案当作最后的预测答案
            except RuntimeError:
                continue
            else:
                if idx_1.item() == answer_id:
                    acc_num_1 = acc_num_1 + 1
        return acc_num_1 / all_num

    def lr_lambda_fun(self, current_iteration: int) -> float:
        """Returns a learning rate multiplier.

        Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
        and then gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        current_epoch = float(current_iteration) / self.iterations
        if current_epoch <= self.__option.config["solver"]["warmup_epochs"]:
            alpha = current_epoch / float(self.__option.config["solver"]["warmup_epochs"])
            return self.__option.config["solver"]["warmup_factor"] * (1. - alpha) + alpha
        else:
            idx = bisect(self.__option.config["solver"]["lr_milestones"], current_epoch)
            return pow(self.__option.config["solver"]["lr_gamma"], idx)

    def get_dataset(self):
        train_dataloader = DataLoader(self.__train_dataset,
                                      batch_size=self.__option.config['solver']['batch_size'],
                                      num_workers=4,
                                      shuffle=True,
                                      collate_fn=collate_fn)
        test_dataloader = DataLoader(self.__test_dataset,
                                     batch_size=self.__option.config['solver']['batch_size'],
                                     num_workers=4,
                                     shuffle=True,
                                     collate_fn=collate_fn)
        return self.__train_dataset, train_dataloader, test_dataloader

    def train(self):
        # hues.info(self.__option.config["solver"]["lr_milestones"], type(self.__option.config["solver"]["lr_milestones"]));input('s')
        train_datalen, train_dataloader, test_dataloader = self.get_dataset()
        #self.__model = self.__model.to(self.__device)
        self.__model.train()
        self.iterations = len(train_datalen) // self.__option.config["solver"]["batch_size"] + 1
        self.__scheduler = lr_scheduler.LambdaLR(self.__optimizer, lr_lambda=self.lr_lambda_fun)
        T = self.iterations * (self.__option.config["solver"]["num_epochs"] -
                               self.__option.config["solver"]["warmup_epochs"] + 1)
        scheduler2 = lr_scheduler.CosineAnnealingLR(
            self.__optimizer, int(T), eta_min=self.__option.config["solver"]["eta_min"], last_epoch=-1)

        # 定期保存模型文件
        checkpoint_manager = CheckpointManager(self.__model,
                                               self.__optimizer,
                                               self.__save_path,
                                               config=self.__option.config)

        global_iteration_step = self.__start_epoch * self.iterations
        acc_1 = -1.0
        train_acc = -1.0
        acc_his = []
        train_his = []
        
        for epoch in range(self.__start_epoch, self.__num_epoch):
            print(f"\n Epoch {epoch} / {self.__num_epoch}")

            train_answers = []  # 保存真实答案
            train_preds = []  # 保存预测答案

            for i, batch in enumerate(tqdm(train_dataloader)):
                self.__optimizer.zero_grad()
                fact_batch_graph, ques_state_weight_softmax = self.__model(batch)
                total_loss = Operation.cal_batch_loss(fact_batch_graph, batch, self.__device,
                                                      loss_fn=self.__loss_function, ques_state_weight_softmax=ques_state_weight_softmax)  # CrossEntropyLoss
                total_loss.backward()
                # hues.info(self.__model.get_pa[0].weight.grad, self.__model.get_pa[2].weight.grad)
                # pred_weight = {name: param.data for name, param in self.__model.named_parameters()}
                self.__optimizer.step()
                # temp_weight = {name: param.data for name, param in self.__model.named_parameters()}

                ''' 输出过程信息 '''
                # # @TODO 输出模型参数信息
                # modelDict = {name: param.data for name, param in self.__model.named_parameters()}
                # # hues.info(modelDict.keys());input('s')
                # if i != 0:
                #     hues.info(weight_perdatt.equal(modelDict['fact_node_att_proj_node.weight']))
                #     hues.info('必须有变化的：', weight_predmlp0.equal(modelDict['mlp.0.weight']))
                #     hues.info(weight_pred0.equal(modelDict['get_pa.0.weight']))
                #     hues.info(weight_pred2.equal(modelDict['get_pa.2.weight']))
                # weight_perdatt = modelDict['fact_node_att_proj_node.weight']
                # weight_predmlp0 = modelDict['mlp.0.weight']
                # weight_pred0 = modelDict['get_pa.0.weight']
                # weight_pred2 = modelDict['get_pa.2.weight']

                print(
                    f" loss {total_loss.item()} | temp epoch {epoch} | train acc {train_his} |\
                    inference step {[i.item() for i in torch.argmax(ques_state_weight_softmax.squeeze(dim=-1), dim=1)]}")

                fact_graphs = dgl.unbatch(fact_batch_graph)
                for i, fact_graph in enumerate(fact_graphs):  # 从batch中拆分出item
                    train_pred = fact_graph.ndata['h']  # (num_nodes,2)
                    train_preds.append(train_pred[:, 1])  # [(num_nodes,)]
                    train_answers.append(batch['facts_answer_id_list'][i])

                # 学习率更新
                if global_iteration_step <= self.iterations * self.__option.config["solver"][
                    "warmup_epochs"]:
                    self.__scheduler.step(global_iteration_step)
                else:
                    global_iteration_step_in_2 = self.iterations * self.__option.config["solver"][
                        "warmup_epochs"] + 1 - global_iteration_step
                    scheduler2.step(int(global_iteration_step_in_2))

                global_iteration_step = global_iteration_step + 1
                torch.cuda.empty_cache()

            # 保存模型参数， 计算acc
            checkpoint_manager.step(epoch)
            train_acc = Operation.cal_acc(train_answers, train_preds)
            train_his.append(train_acc)
            print(" | train_acc {:.2%} ".format(train_acc), end='')
            if (epoch % 500 == 0) and False:
                self.__model.eval()
                preds = []
                answers = []
                # val
                for i, batch in enumerate(test_dataloader):  # 查看一个batch的acc
                    with torch.no_grad():
                        fact_batch_graph, inference_step, max_inference_step = self.__model(batch)

                    fact_graphs = dgl.unbatch(fact_batch_graph)

                    for i, fact_graph in enumerate(fact_graphs):
                        pred = fact_graph.ndata['h']  # (num_nudes, 2)
                        preds.append(pred[:, 1])  # 下标为1表示作为答案的概率
                        answers.append(batch['facts_answer_id_list'][i])

                # calculate top@1,top@3
                acc_1 = Operation.cal_acc(answers, preds)
                print(" | val_acc {:.2%}  \n".format(acc_1))
                acc_his.append(acc_1)
                print(acc_his)
                print('\n')
                self.__model.train()
            torch.cuda.empty_cache()