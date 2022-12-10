import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as numpy
from util.dynamic_rnn import DynamicRNN

from model.img_gcn import ImageGCN
from model.semantic_gcn import SemanticGCN
from model.fact_gcn import FactGCN

from model.global_gcn import GlobalGCN
from model.RIM import RIM
from model.RIM import RIMCell
from model.memory_network import MemoryNetwork, MemoryGate
from model.Attention import AdditiveAttention

import dgl
import networkx as nx
import numpy as np
import hues


class CMGCNnet_RIM(nn.Module):
    active_his = None

    def __init__(self, config, que_vocabulary, glove, device, model_type=1):
        '''
        :param config: 配置参数
        :param que_vocabulary: 字典 word 2 index
        :param glove: (voc_size,embed_size)
        '''
        super(CMGCNnet_RIM, self).__init__()
        self.model_type = model_type
        if self.model_type == 1:
            self.num_unit = 8
        elif self.model_type == 2:
            self.num_unit = 3
        elif self.model_type == 3:
            self.num_unit = 5
        elif self.model_type == 4:
            self.num_unit = 6
        elif self.model_type == 5:
            self.num_unit = 8
        elif self.model_type == 6:
            self.num_unit = 8
        self.config = config
        self.device = device
        # 构建question glove嵌入层
        self.que_glove_embed = nn.Embedding(len(que_vocabulary), config['model']['glove_embedding_size'])
        # 读入初始参数
        self.que_glove_embed.weight.data = glove
        # 固定初始参数
        self.que_glove_embed.weight.requires_grad = False

        # 问题嵌入lstm
        self.ques_rnn = nn.LSTM(config['model']['glove_embedding_size'],
                                config['model']['lstm_hidden_size'],
                                config['model']['lstm_num_layers'],
                                batch_first=True,
                                dropout=config['model']['dropout'])
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # question guided visual node attention
        self.vis_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)

        # question guided visual relation attention
        self.vis_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_proj_rel = nn.Linear(
            config['model']['vis_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided semantic node attention
        self.sem_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_proj_sem = nn.Linear(
            config['model']['sem_node_dims'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_value = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims'], 1)

        # question guided semantic relation attention
        self.sem_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_proj_rel = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)

        # question guided fact node attention
        self.fact_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_proj_node = nn.Linear(
            config['model']['fact_node_dims'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_value = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims'], 1)

        # image gcn1
        self.img_gcn1 = ImageGCN(config,
                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['vis_relation_dims'])

        # semantic gcn1
        self.sem_gcn1 = SemanticGCN(config,
                                    in_dim=config['model']['sem_node_dims'],
                                    out_dim=config['model']['semantic_gcn1_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])
        # fact gcn1
        self.fact_gcn1 = FactGCN(config,
                                 in_dim=config['model']['fact_node_dims'],
                                 out_dim=config['model']['fact_gcn1_out_dim'])

        self.visual_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                   memory_size=config['model']['image_gcn1_out_dim'],
                                                   que_szie=config['model']['lstm_hidden_size'],
                                                   query_hidden_size=config['model']['visual_memory_query_hidden_size'],
                                                   memory_relation_size=config['model']['vis_relation_dims'],
                                                   memory_hidden_size=config['model'][
                                                       'visual_memory_memory_hidden_size'],
                                                   mem_read_att_proj=config['model'][
                                                       'visual_memory_memory_read_att_size'],
                                                   T=config['model']['memory_step'])

        self.semantic_memory_network = MemoryNetwork(query_input_size=config['model']['fact_gcn1_out_dim'],
                                                     memory_size=config['model']['semantic_gcn1_out_dim'],
                                                     que_szie=config['model']['lstm_hidden_size'],
                                                     query_hidden_size=config['model'][
                                                         'semantic_memory_query_hidden_size'],
                                                     memory_relation_size=config['model']['sem_relation_dims'],
                                                     memory_hidden_size=config['model'][
                                                         'semantic_memory_memory_hidden_size'],
                                                     mem_read_att_proj=config['model'][
                                                         'semantic_memory_memory_read_att_size'],
                                                     T=config['model']['memory_step'])

        self.memory_gate = MemoryGate(vis_mem_size=config['model']['visual_memory_query_hidden_size'],
                                      sem_mem_size=config['model']['semantic_memory_query_hidden_size'],
                                      node_size=config['model']['fact_gcn1_out_dim'],
                                      out_size=config['model']['memory_gate_out_dim'])

        self.global_gcn = GlobalGCN(config, in_dim=512, out_dim=512)

        self.mlp = nn.Sequential(
            nn.Linear(config['model']['memory_gate_out_dim'] + config['model']['lstm_hidden_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

        # RIM LAYER option
        if self.model_type == 1:
            hues.info('loading action1 8_4')
            self.rim_cell_action1 = RIMCell(self.device, input_size=300, hidden_size=100, num_units=8, k=4,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(1612, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        elif self.model_type == 2:
            hues.info('loading action2 3_2')
            self.rim_cell_action1 = RIMCell(self.device, input_size=300, hidden_size=100, num_units=3, k=2,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(1112, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        elif self.model_type == 3:
            hues.info('loading action3 5_3')
            self.rim_cell_action1 = RIMCell(self.device, input_size=300, hidden_size=100, num_units=5, k=3,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(1312, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        elif self.model_type == 4:
            hues.info('loading action4 6_2')
            self.rim_cell_action1 = RIMCell(self.device, input_size=300, hidden_size=100, num_units=6, k=2,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(1412, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
        elif self.model_type == 5:
            hues.info('loading action5 attention 8_4')
            self.mem_attention = AdditiveAttention(key_size=300, query_size=512, num_hiddens=300, dropout=0.5)
            self.rim_cell_action1 = RIMCell(self.device, input_size=812, hidden_size=64, num_units=8, k=4,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(812, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
            self.ques_binary = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        elif self.model_type == 6:
            hues.info('loading model_type6 with action5')
            self.mem_attention = AdditiveAttention(key_size=300, query_size=512, num_hiddens=300, dropout=0.5)
            self.rim_cell_action1 = RIMCell(self.device, input_size=812, hidden_size=64, num_units=8, k=4,
                                            rnn_cell="LSTM")
            self.mlp_action1 = nn.Sequential(
                nn.Linear(812, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
            self.ques_binary = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(self, batch):
        # ======================================================================================
        #                                    数据处理
        # ======================================================================================

        batch_size = len(batch['id_list'])

        # image
        images = batch['features_list']  # [(36,2048)]
        images = torch.stack(images).to(self.device)  # (batch,36,2048)

        img_relations = batch['img_relations_list']
        img_relations = torch.stack(img_relations).to(self.device)  # (batch,36,36,7)

        # question
        questions = batch['question_list']  # [(max_length,)]
        questions = torch.stack(questions).to(self.device)  # (batch,max_length)
        # 15: 54:14 - INFO - tensor(3, device='cuda:3')
        # 15: 54:14 - INFO - tensor(7, device='cuda:3')
        # 15: 54:14 - INFO - tensor(48, device='cuda:3')
        # hues.info(questions[0][0])
        # hues.info(questions[0][1])
        # hues.info(questions[0][2])
        # input('s')

        questions_len_list = batch['question_length_list']
        questions_len_list = torch.Tensor(questions_len_list).long().to(self.device)

        # semantic graph
        semantic_num_nodes_list = torch.Tensor(batch['semantic_num_nodes_list']).long().to(self.device)

        semantic_n_features_list = batch['semantic_node_features_list']
        semantic_n_features_list = [features.to(self.device) for features in semantic_n_features_list]

        semantic_e1ids_list = batch['semantic_e1ids_list']
        semantic_e1ids_list = [e1ids.to(self.device) for e1ids in semantic_e1ids_list]

        semantic_e2ids_list = batch['semantic_e2ids_list']
        semantic_e2ids_list = [e2ids.to(self.device) for e2ids in semantic_e2ids_list]

        semantic_e_features_list = batch['semantic_edge_features_list']
        semantic_e_features_list = [features.to(self.device) for features in semantic_e_features_list]

        # fact graph
        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list']).long().to(self.device)

        facts_features_list = batch['facts_node_features_list']
        facts_features_list = [features.to(self.device) for features in facts_features_list]

        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [e1ids.to(self.device) for e1ids in facts_e1ids_list]

        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [e2ids.to(self.device) for e2ids in facts_e2ids_list]

        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [answer.to(self.device) for answer in facts_answer_list]

        # 初始化权重

        # ===============================================================================================================
        #                               1. embed questions
        # ===============================================================================================================
        ques_embed = self.que_glove_embed(questions).float()  # shape (batch,question_max_length,300)

        # 这里用最后一个LSTM单元隐层的输出hn当做句子的表示
        _, (ques_embed, _) = self.ques_rnn(ques_embed, questions_len_list)  # qes_embed shape=(batch,hidden_size)

        # ===============================================================================================================
        #                               2. question guided visual node attention
        # ===============================================================================================================
        # ques_embed : 20:48:45 - INFO - torch.Size([11, 512])
        # images : 20:48:45 - INFO - torch.Size([11, 36, 2048])
        node_att_proj_ques_embed = self.vis_node_att_proj_ques(ques_embed)  # shape (batch,proj_size)
        node_att_proj_img_embed = self.vis_node_att_proj_img(images)  # shape (batch,36,proj_size)
        # repeat 为了和image有相同的维数36
        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1],
                                                                                1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)

        # ===============================================================================================================
        #                                3. question guided visual relation attention
        # ===============================================================================================================
        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques(ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel(img_relations)  # shape(batch,36,36,128)
        # 改变question的维度
        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        for i in range(batch_size):
            # ===============================================================================================================
            #                                4 question guided semantic node attention
            # ===============================================================================================================
            num_node = semantic_num_nodes_list[i]  # n
            sem_node_features = semantic_n_features_list[i]  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques(q_embed)  # shape (n,p)
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem(sem_node_features)  # shape (n,p)
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed)  # shape (n,p)
            sem_node_att_values = self.sem_node_att_value(sem_node_att_proj_sem_sum_ques)  # shape(n,1)
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  # shape(n,1)

            sem_node_att_val_list.append(sem_node_att_values)

            # ===============================================================================================================
            #                                5 question guided semantic relation attention
            # ===============================================================================================================
            num_edge = semantic_e_features_list[i].shape[0]  # n
            sem_edge_features = semantic_e_features_list[i]  # (n,300)
            qq_embed = ques_embed[i]  # (512)
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  # (n,512)
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques(qq_embed)  # shape (n,p)
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel(sem_edge_features)  # shape (n,p)
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  # shape (n,p)
            sem_rel_att_values = self.sem_rel_att_value(sem_rel_att_proj_rel_sum_ques)  # shape(n,1)
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  # shape(n,1)

            sem_edge_att_val_list.append(sem_rel_att_values)

        # ===============================================================================================================
        #                                6 question guided fact node attention
        # ===============================================================================================================
        fact_node_att_values_list = []
        for i in range(batch_size):
            num_node = fact_num_nodes_list[i]  # n
            fact_node_features = facts_features_list[i]  # (n,1024)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)
            fact_node_att_values_list.append(fact_node_att_values)

        # ===============================================================================================================
        #                             7 Build Image Graph
        # ===============================================================================================================
        # 建图 36 nodes,36*36 edges
        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()
            g = g.to(self.device)  # 将图加入到cuda中
            # add nodes
            g.add_nodes(36)
            # add node features
            g.ndata['h'] = images[i]
            g.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)
            g.ndata['batch'] = torch.full([36, 1], i).to(self.device)  # 加入cuda中
            # add edges
            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)
            # add edge features
            g.edata['rel'] = img_relations[i].view(36 * 36, self.config['model']['vis_relation_dims'])  # shape(36*36,7)
            g.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)  # shape(36*36,1)
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)

        # ===============================================================================================================
        #                                8 Build Semantic Graph
        # ===============================================================================================================
        semantic_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(semantic_num_nodes_list[i])
            graph.add_edges(semantic_e1ids_list[i], semantic_e2ids_list[i])
            graph.ndata['h'] = semantic_n_features_list[i]  # 结点的表示
            graph.ndata['att'] = sem_node_att_val_list[i]
            graph.edata['rel'] = semantic_e_features_list[i]  # 边的表示（关系的表示）
            graph.edata['att'] = sem_edge_att_val_list[i]
            semantic_graphs.append(graph)
        semantic_batch_graph = dgl.batch(semantic_graphs)

        # ===============================================================================================================
        #                                9 Build Fact Graph
        # ===============================================================================================================
        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph = graph.to(self.device)
            graph.add_nodes(fact_num_nodes_list[i])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]  # 结点的表示
            graph.ndata['att'] = fact_node_att_values_list[i]
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i).to(self.device)
            graph.ndata['answer'] = facts_answer_list[i]
            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)

        # ===============================================================================================================
        #                                8. Intra GCN
        # ===============================================================================================================
        # (1). 对visual graph做 gcn
        image_batch_graph = self.img_gcn1(image_batch_graph)
        # (2) 对semantic graph做 gcn
        semantic_batch_graph = self.sem_gcn1(semantic_batch_graph)
        # (2) 对 fact graph做 gcn
        fact_batch_graph = self.fact_gcn1(fact_batch_graph)
        fact_batch_graph.ndata['hh'] = fact_batch_graph.ndata['h']

        # ===============================================================================================================
        #                                9. Memory network
        # ===============================================================================================================
        image_graph_list = dgl.unbatch(image_batch_graph)
        semantic_graph_list = dgl.unbatch(semantic_batch_graph)
        fact_graph_list = dgl.unbatch(fact_batch_graph)
        new_fact_graph_list = []
        for i, fact_graph in enumerate(fact_graph_list):
            question = ques_embed[i]

            num_fact_nodes = fact_graph.number_of_nodes()
            image_graph = image_graph_list[i]
            semantic_graph = semantic_graph_list[i]

            question = ques_embed[i]
            fact_graph_memory_visual = self.visual_memory_network(fact_graph, image_graph, question)
            fact_graph_memory_semantic = self.semantic_memory_network(fact_graph, semantic_graph, question)
            fact_graph.ndata['vis_mem'] = fact_graph_memory_visual.ndata['h']
            fact_graph.ndata['sem_mem'] = fact_graph_memory_semantic.ndata['h']
            # 每个fact graph的结点拼接上question
            fact_graph.ndata['hh_cat'] = torch.cat(
                (fact_graph.ndata['hh'], question.unsqueeze(0).repeat(fact_graph.ndata['hh'].shape[0], 1)), dim=1)
            new_fact_graph_list.append(fact_graph)

        ''' RIM处理 '''
        new_fact_batch_graph = dgl.batch(new_fact_graph_list)
        new_fact_batch_graph.ndata['rim_h'], ques_state_weight_softmax = self.action_5(new_fact_batch_graph, ques_embed)

        ''' MLP二分类 '''
        new_fact_batch_graph.ndata['h'] = self.mlp_action1(new_fact_batch_graph.ndata['rim_h'])
        return new_fact_batch_graph, ques_state_weight_softmax

    # return 结果送入直接MLP二分类
    '''
            1.1 将Mem中加入空白记忆，如果attention机制选择了空白记忆，则退出推理
            1.2 加入二分类机制，判断当前question state是否足够得出答案, 如果足够得出答案，则退出推理
                1.2.1 新的损失函数：loss_fn + inference/max_inference
            1.3 如果softmax之前的attention都比较低，低于一个阈值，则跳出推理
            2. rim的结果为 512，拼接上fact 300为812，后送入mlp二分类
    '''

    def action_2(self, new_fact_batch_graph, ques_embed):
        # hh : 原始Fact node的embedding
        # vis_mem / sem_mem : 两种mem   三种数据的长度相同
        # attention layer = key_size00,=3 query_size=512, num_hiddens=300
        # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
        # key / value : [batch_size, node_num_sum, 300]
        g_list = dgl.unbatch(new_fact_batch_graph)
        batch_size = len(g_list)
        max_node_num = 0  # batch中，每个item的node_num的最大值
        enc_valid_lens = torch.zeros(batch_size).to(self.device)  # batch中，每个item的Mem的有效长度
        for i, g in enumerate(g_list):
            enc_valid_lens[i] = g.ndata['hh'].shape[0] * 3
            if max_node_num < g.ndata['hh'].shape[0]:
                max_node_num = g.ndata['hh'].shape[0]
        ques_embed = torch.unsqueeze(ques_embed, dim=1).to(self.device)

        ' 设置Mem， 长度为max_node_num, 有效长度由enc_valid_lens指出 '
        Mem = torch.zeros(batch_size, max_node_num * 3, 300).to(self.device)
        for i, g in enumerate(g_list):
            vis_mem = g.ndata['vis_mem']
            sem_mem = g.ndata['sem_mem']
            fac_nod = g.ndata['hh']
            cat = torch.cat((vis_mem, sem_mem, fac_nod), dim=0)

            for j in range(cat.shape[0]):
                Mem[i][j] = cat[j]

        ' RIM 输出参数设置 '
        hs = torch.randn(batch_size, self.num_unit, 64).to(self.device)
        cs = torch.randn(batch_size, self.num_unit, 64).to(self.device)

        ' 推理过程 '
        self.inference_step = 0
        self.max_inference_step = max_node_num * 3
        question_state = ques_embed
        while self.inference_step < max_node_num * 3:
            ' index得到的是 batch中每个item当前步骤需要激活的mem '
            index = self.mem_attention(question_state, Mem, Mem, enc_valid_lens)
            # active_mem_batch = torch.zeros(batch_size, 1, 300).to(self.device)
            # for i in range(batch_size):
            #     active_mem_batch[i][0] = Mem[i][index[i]]
            ' active_mem_batch 是 batch中的每个item所需要的mem  shape = [batch_size, 1, 300] '
            ' rim_x 是每个batch的item选出的mem去拼接对应问题状态的结果，用于输入到rim中 '
            rim_x = torch.zeros(batch_size, 1, 812).to(self.device)
            for i in range(batch_size):
                rim_x[i][0] = torch.cat((Mem[i][index[i]], question_state[i][0]), dim=0)
            hs, cs = self.rim_cell_action1(rim_x, hs, cs)
            question_state = hs.contiguous().view(batch_size, -1)
            # hues.info(ques_embed.shape)  [11, 512]
            self.inference_step = self.inference_step + 1
            judge = torch.argmax(self.ques_state(question_state), dim=1).to(self.device)
            if judge.equal(torch.ones(batch_size).to(self.device)):
                break
            question_state = question_state.unsqueeze(dim=1)  # shape = [batch_size, 1, 512]

        ' 得到问题当前状态 question_state.shape = [batch_size, 512] 拼接到候选答案中'
        fact_node_cat_rim = torch.zeros(new_fact_batch_graph.ndata['hh'].shape[0], 300 + 512).to(self.device)
        
        for i in range(new_fact_batch_graph.ndata['hh'].shape[0]):
            temp_fact_h = new_fact_batch_graph.ndata['hh'][i]
            temp_question_state = question_state[new_fact_batch_graph.ndata['batch'][i].item()].squeeze(dim=0)
            fact_node_cat_rim[i] = torch.cat((temp_fact_h, temp_question_state))

        return fact_node_cat_rim

    ' test方法需要知道每次都有哪个mem激活了，做出heatmap '
    def action_2_test_method(self, new_fact_batch_graph, ques_embed):
        self.active_his = torch.zeros(batch_size, max_node_num * 3, 1)  # 保存item中每个结点被激活的次数

    ' 多输出结构，权重控制偏向推理步骤还是准确率 '
    def action_3(self, new_fact_batch_graph, ques_embed):
        # hh : 原始Fact node的embedding
        # vis_mem / sem_mem : 两种mem   三种数据的长度相同
        # attention layer = key_size00,=3 query_size=512, num_hiddens=300
        # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
        # key / value : [batch_size, node_num_sum, 300]
        g_list = dgl.unbatch(new_fact_batch_graph)
        batch_size = len(g_list)
        max_node_num = 0  # batch中，每个item的node_num的最大值
        enc_valid_lens = torch.zeros(batch_size).to(self.device)  # batch中，每个item的Mem的有效长度
        for i, g in enumerate(g_list):
            enc_valid_lens[i] = g.ndata['hh'].shape[0] * 3
            if max_node_num < g.ndata['hh'].shape[0]:
                max_node_num = g.ndata['hh'].shape[0]
        ques_embed = torch.unsqueeze(ques_embed, dim=1).to(self.device)

        ' 设置Mem， 长度为max_node_num, 有效长度由enc_valid_lens指出 '
        Mem = torch.zeros(batch_size, max_node_num * 3, 300).to(self.device)
        for i, g in enumerate(g_list):
            vis_mem = g.ndata['vis_mem']
            sem_mem = g.ndata['sem_mem']
            fac_nod = g.ndata['hh']
            cat = torch.cat((vis_mem, sem_mem, fac_nod), dim=0)

            for j in range(cat.shape[0]):
                Mem[i][j] = cat[j]

        ' RIM 输出参数设置 '
        hs = torch.randn(batch_size, self.num_unit, 64).to(self.device)
        cs = torch.randn(batch_size, self.num_unit, 64).to(self.device)

        ' 推理过程 '
        question_state = ques_embed
        question_state_his = torch.zeros(batch_size, max_node_num*3, 512).to(self.device)
        for i in range(max_node_num * 3):
            ' M_a得到的是每个batch中 attention后的memory '
            M_a = self.mem_attention(question_state, Mem, Mem, enc_valid_lens)  # size = [11, 1, 300]

            hs, cs = self.rim_cell_action1(torch.cat((M_a, question_state), dim=2), hs, cs)
            question_state = hs.contiguous().view(batch_size, -1)  # shape = [batch_size, 512]
            question_state_his[:, i, :] = question_state
            question_state = question_state.unsqueeze(dim=1)

        #网络part 2 根据question_state
        ques_state_weight = self.ques_binary(question_state_his)  # shape = [11, 100, 2]
        ques_state_weight_softmax = torch.softmax(ques_state_weight[:,:,1].squeeze(dim=-1), dim=1).unsqueeze(dim=-1)  # shape = [11, 100, 1]
        ques_state_weight_softmax_T = torch.zeros(ques_state_weight_softmax.shape[0], ques_state_weight_softmax.shape[2], ques_state_weight_softmax.shape[1]).to(self.device)
        for i, item in enumerate(ques_state_weight_softmax):
            ques_state_weight_softmax_T[i] = item.t()
        question_state = torch.bmm(ques_state_weight_softmax_T, question_state_his)  # shape = [11, 1, 512]

        ' 得到问题当前状态 question_state.shape = [batch_size, 512] 拼接到候选答案中'
        fact_node_cat_rim = torch.zeros(new_fact_batch_graph.ndata['hh'].shape[0], 300 + 512).to(self.device) # shape = [sum(fact_node), 812]
        for i in range(new_fact_batch_graph.ndata['hh'].shape[0]):  # for 每个节点
            temp_fact_h = new_fact_batch_graph.ndata['hh'][i]
            temp_question_state = question_state[new_fact_batch_graph.ndata['batch'][i].item()].squeeze(dim=0)  # 每个节点所属batch的q_state
            fact_node_cat_rim[i] = torch.cat((temp_fact_h, temp_question_state))

        return fact_node_cat_rim, ques_state_weight
    
    ' 与 action_3 相比，更改了ques_state权重生成，现在生成一个分布概率，去拟合一个高斯分布，同时记录最高权重值的平均值，以供test阶段选择结束位置 '
    def action_5(self, new_fact_batch_graph, ques_embed):
        # hh : 原始Fact node的embedding
        # vis_mem / sem_mem : 两种mem   三种数据的长度相同
        # attention layer = key_size00,=3 query_size=512, num_hiddens=300
        # Shape of `query`: (`batch_size`, 1, `num_hiddens`)
        # key / value : [batch_size, node_num_sum, 300]
        g_list = dgl.unbatch(new_fact_batch_graph)
        batch_size = len(g_list)
        max_node_num = 0  # batch中，每个item的node_num的最大值
        enc_valid_lens = torch.zeros(batch_size).to(self.device)  # batch中，每个item的Mem的有效长度
        for i, g in enumerate(g_list):
            enc_valid_lens[i] = g.ndata['hh'].shape[0] * 3
            if max_node_num < g.ndata['hh'].shape[0]:
                max_node_num = g.ndata['hh'].shape[0]
        ques_embed = torch.unsqueeze(ques_embed, dim=1).to(self.device)

        ' 设置Mem， 长度为max_node_num, 有效长度由enc_valid_lens指出 '
        Mem = torch.zeros(batch_size, max_node_num * 3, 300).to(self.device)
        for i, g in enumerate(g_list):
            vis_mem = g.ndata['vis_mem']
            sem_mem = g.ndata['sem_mem']
            fac_nod = g.ndata['hh']
            cat = torch.cat((vis_mem, sem_mem, fac_nod), dim=0)

            for j in range(cat.shape[0]):
                Mem[i][j] = cat[j]

        ' RIM 输出参数设置 '
        hs = torch.randn(batch_size, self.num_unit, 64).to(self.device)
        cs = torch.randn(batch_size, self.num_unit, 64).to(self.device)

        ' 推理过程 '
        question_state = ques_embed
        question_state_his = torch.zeros(batch_size, max_node_num*3, 512).to(self.device)
        for i in range(max_node_num * 3):
            ' M_a得到的是每个batch中 attention后的memory '
            M_a = self.mem_attention(question_state, Mem, Mem, enc_valid_lens)  # size = [11, 1, 300]

            hs, cs = self.rim_cell_action1(torch.cat((M_a, question_state), dim=2), hs, cs)
            question_state = hs.contiguous().view(batch_size, -1)  # shape = [batch_size, 512]
            question_state_his[:, i, :] = question_state
            question_state = question_state.unsqueeze(dim=1)
 
        #网络part 2
        ques_state_his_detached = question_state_his.detach()  # 在此部分切断梯度信息
        ques_state_weight = self.ques_binary(ques_state_his_detached)  # shape = [11, 100, 1]
        ques_state_weight_softmax = torch.softmax(ques_state_weight.squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)  # shape = [11, 100, 1]
        ques_state_weight_softmax_T = torch.zeros(ques_state_weight_softmax.shape[0], ques_state_weight_softmax.shape[2], ques_state_weight_softmax.shape[1]).to(self.device)
        for i, item in enumerate(ques_state_weight_softmax):
            ques_state_weight_softmax_T[i] = item.t()
        question_state = torch.bmm(ques_state_weight_softmax_T, question_state_his)  # shape = [11, 1, 512]

        ' 得到问题当前状态 question_state.shape = [batch_size, 512] 拼接到候选答案中'
        fact_node_cat_rim = torch.zeros(new_fact_batch_graph.ndata['hh'].shape[0], 300 + 512).to(self.device) # shape = [sum(fact_node), 812]
        for i in range(new_fact_batch_graph.ndata['hh'].shape[0]):  # for 每个节点
            temp_fact_h = new_fact_batch_graph.ndata['hh'][i]
            temp_question_state = question_state[new_fact_batch_graph.ndata['batch'][i].item()].squeeze(dim=0)  # 每个节点所属batch的q_state
            fact_node_cat_rim[i] = torch.cat((temp_fact_h, temp_question_state))

        return fact_node_cat_rim, ques_state_weight_softmax

    ' 如果选择attention数值低于多少的就退出循环 或者 总体attention低于多少就退出'
    def action_4(self, new_fact_batch_graph, ques_embed):
        pass

    # self.rim_cell_action1 = RIMCell(self.device, input_size=300, hidden_size=100, num_units=8, k=4, rnn_cell="LSTM")
    # 参照LSTM的实现方式, 计算出最后的RIM输出h，
    # 先计算，后拼question
    ' 使用action1 修改forward方法return内容  修改train中接收model返回值 修改val中接收model返回值 修改计算batch loss的方法 '
    def action_1(self, new_fact_batch_graph, ques_embed):
        # x = [batch_size, node, embedding]
        # ques_embed.shape = [batch_size, 512]
        g_list = dgl.unbatch(new_fact_batch_graph)
        batch_size = len(g_list)
        max_node_num = 0
        for g in g_list:
            if max_node_num < g.ndata['hh'].shape[0]:
                max_node_num = g.ndata['hh'].shape[0]
        x_vis = torch.zeros(batch_size, max_node_num, 300)
        x_sem = torch.zeros(batch_size, max_node_num, 300)
        x_fac = torch.zeros(batch_size, max_node_num, 300)
        for i, g in enumerate(g_list):
            vis_mem = g.ndata['vis_mem']
            sem_mem = g.ndata['sem_mem']
            fac_nod = g.ndata['hh']
            for j in range(0, fac_nod.shape[0]):
                x_vis[i, j] = vis_mem[j]
                x_sem[i, j] = sem_mem[j]
                x_fac[i, j] = fac_nod[j]
        x = torch.cat((x_vis, x_sem, x_fac), dim=1).float()  # x.shape = [batch_size, max_node_num*3, 300]
        xs = torch.split(x, 1, 1)  # len(xs) = max_node_num*3
        # xs[1].shape = [batch_size, 1, 300]
        hs = torch.randn(x.size(0), self.num_unit, 100).to(self.device)
        cs = torch.randn(x.size(0), self.num_unit, 100).to(self.device)
        
        for x in xs:
            x = x.to(self.device)  # x.shape = [batch_size, 1, 300]
            hs, cs = self.rim_cell_action1(x, hs, cs)
        rim_h = hs.contiguous().view(x.size(0), -1)  # h.view = [16, num_unit * 100]  一个batch中有16个问题，每个问题对应一个h
        # hues.info(new_fact_batch_graph.ndata['batch'].shape) # shape = [836, 1]  当前结点的batch数目

        # fact node的embedding hh拼接上得出的rim_h 再拼接上question
        fact_node_cat_rim = torch.zeros(new_fact_batch_graph.ndata['hh'].shape[0], rim_h.shape[1] + 512 + 300).to(
            self.device)

        for i in range(new_fact_batch_graph.ndata['hh'].shape[0]):
            temp_fact_h = new_fact_batch_graph.ndata['hh'][i]  # 获取第i个node的embedding
            temp_batch_rim_h = rim_h[len(new_fact_batch_graph.ndata['batch'][i])]  # 获取第i个node所属question的rim_h
            temp_question = ques_embed[len(new_fact_batch_graph.ndata['batch'][i])]  # 获得第i个node所属问题的 question embedding
            cat = torch.cat((temp_fact_h, temp_question, temp_batch_rim_h))
            fact_node_cat_rim[i] = cat

        return fact_node_cat_rim  # shape = [551, 1612]
