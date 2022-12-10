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
from model.model_rim_okvqa import CMGCNnet_RIM
from model.model_baseline import CMGCNnet
from model.model_test_1 import MMR_Test

def collate_fn(batch):
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

    question_type_list = []

    for item in batch:
        # question
        qid = item['id']
        qid_list.append(qid)

        question = item['question']
        question_list.append(question)

        question_length = item['question_length']
        question_length_list.append(question_length)

        question_type_list.append(item['question_type'])

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
    res['question_type_list'] = question_type_list
    return res

class Test:
    __device = None
    __k = 1
    __modelPath = None

    __dataset = None
    __tag = None  # FVQA or OKVQA
    __option = None

    que_types_dict = {"eight": 0, "nine": 0, "four": 0, "six": 0, "two": 0,
                          "other": 0, "one": 0, "five": 0, "ten": 0, "seven": 0, "three": 0}
    que_types_res_dict = {"eight": 0, "nine": 0, "four": 0, "six": 0, "two": 0,
                            "other": 0, "one": 0, "five": 0, "ten": 0, "seven": 0, "three": 0}

    def __init__(self, device, model_path, model, k=1):
        self.__device = device
        self.__k = k
        self.__modelPath = model_path
        self.__model = model
        self.model_type=6

    def setDataset(self, dataset, tag):
        self.__dataset = dataset
        self.__tag = tag

    def setOption(self, option):
        self.__option = option

    def get_dataset(self):
        test_dataloader = DataLoader(self.__dataset,
                                     batch_size=self.__option.config['solver']['batch_size'],
                                     num_workers=4,
                                     shuffle=True,
                                     collate_fn=collate_fn)
        return test_dataloader

    def startTest(self):
        if self.__tag == 'OKVQA':
            self.OKVQATest()
        elif self.__tag == 'FVQA':
            pass

    def OKVQATest(self):
        glovevocabulary = Vocabulary(self.__option.config["dataset"]["word_counts_json"],
                                     min_count=self.__option.config["dataset"]["vocab_min_count"])
        glove = np.load(self.__option.config['dataset']['glove_vec_path'])
        glove = torch.Tensor(glove)
        if self.__model == 'MMR':
            hues.info('loading MMR model')
            model = CMGCNnet_RIM(self.__option.config,
                                 que_vocabulary=glovevocabulary,
                                 glove=glove,
                                 device=self.__device,
                                 model_type=self.model_type)
        elif self.__model == 'base':
            hues.info('loading base model')
            model = CMGCNnet(self.__option.config,
                             que_vocabulary=glovevocabulary,
                             glove=glove,
                             device=self.__device)
        elif self.__model == 'Test1':
            hues.info('loading Test1 model')
            model = MMR_Test(self.__option.config,
                                 que_vocabulary=glovevocabulary,
                                 glove=glove,
                                 device=self.__device,
                                 model_type=self.model_type)
        
        model = model.to(self.__device)
        model_state_dict, optimizer_state_dict = load_checkpoint(self.__modelPath)
        model.load_state_dict(model_state_dict)

        model.eval()
        answers = []
        preds = []
        que_types = []
        pa_his = []  # 保存每个问题的推理次数分布  [400*11, 7]

        val_dataloader = self.get_dataset()

        for i, batch in enumerate(tqdm(val_dataloader)):
            for que_type in batch['question_type_list']:
                self.que_types_dict[que_type] = self.que_types_dict[que_type] + 1

            with torch.no_grad():
                fact_batch_graph, pa = model(batch)  # pa.shape = [11, 7, 1]

            pa_his.append(pa)

            fact_graphs = dgl.unbatch(fact_batch_graph)

            for i, fact_graph in enumerate(fact_graphs):
                pred = fact_graph.ndata['h']
                preds.append(pred[:, 1])
                answers.append(batch['facts_answer_id_list'][i])

            que_types = que_types+batch['question_type_list']

        # 保存分布文件
        np.save(file='/home/data/yjgroup/lyl/projects/MMR-VQA/saved_data/pa_his.npy', arr=pa_his)

        # calculate top@1,top@3
        acc_1 = self.cal_acc(answers, preds, que_types=que_types)
        print("acc@1={:.2%}  ".format(acc_1))
        torch.cuda.empty_cache()

        self.cal_type_acc(self.que_types_dict, self.que_types_res_dict)

    @staticmethod
    def cal_type_acc(que_types_dict, que_types_res_dict):
        
        for qt in list(que_types_dict.keys()):
            if que_types_dict[qt] != 0:
                acc = que_types_res_dict[qt] / que_types_dict[qt]
                print(qt, acc*100)


    def cal_acc(self, answers, preds, que_types):
        all_num = len(preds)
        acc_num_1 = 0
        for i, answer_id in enumerate(answers):
            pred = preds[i]  # (num_nodes)
            try:
                # top@1
                _, idx_1 = torch.topk(pred, k=1)

            except RuntimeError:

                continue
            else:
                if idx_1.item() == answer_id:
                    acc_num_1 = acc_num_1 + 1
                    self.que_types_res_dict[que_types[i]] = self.que_types_res_dict[que_types[i]] + 1

        return acc_num_1 / all_num