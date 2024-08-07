import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process

class SubGraphs(nn.Module):
    def __init__(self,input_graph_size,out_graph_size,input_graph_fea,out_graph_fea,batch_size,dropout=0.0):
        super(SubGraphs, self).__init__()

        self.bn = True
        self.act = nn.ReLU()
        self.sub_act = nn.ReLU()
        self.graph_partition = nn.Linear(input_graph_fea,out_graph_size).to("cuda")
        self.dropout = nn.Dropout(dropout)

        self.subgraph_layers = SubgraphGCN_Conv(input_graph_size= input_graph_size,subgraph_num=out_graph_size,input_dim=input_graph_fea,output_dim=out_graph_fea)

        self.subgraph_num = out_graph_size
        self.input_graph_size = input_graph_size
        self.input_graph_fea = input_graph_fea

        self.batch_size = batch_size

    def assignment(self,input_graph,adj_matrix):

        batch_size = input_graph.size()[0]
        s_soft = self.graph_partition(input_graph)
        s_soft = self.act(s_soft)
        s_soft = nn.Softmax(dim=-1)(s_soft)

        s_hard_one_hot = (s_soft==s_soft.max(dim=1,keepdim=True)[0]).to(dtype=torch.float)

        sub_fea_list = []
        sub_adj_list = []
        coarsen_adj = s_hard_one_hot.permute(1,0) @ adj_matrix @ s_hard_one_hot

        l = []
        for i in range(batch_size):
            l.append(torch.diag_embed(torch.ones([self.input_graph_size, ]).to("cuda")).unsqueeze(dim=0))
        self_loop = torch.cat(l, dim=0).to("cuda")

        for i in range(self.subgraph_num):

            sub_remain_nodes = s_hard_one_hot[:,i:(i+1)]
            sub_diag = torch.diag_embed(sub_remain_nodes.squeeze(dim=-1))

            subgraph_fea =  sub_diag @ input_graph
            subgraph_adj = sub_diag @ adj_matrix @ sub_diag

            subgraph_adj = subgraph_adj + self_loop

            sub_fea_list.append(subgraph_fea.unsqueeze(dim=1))
            sub_adj_list.append(subgraph_adj.unsqueeze(dim=1))

        subs_fea = torch.cat(sub_fea_list,dim=1)

        subd_adj = torch.cat(sub_adj_list,dim=1)

        return subs_fea,subd_adj,coarsen_adj,s_hard_one_hot


    def apply_bn(self, x):

        '''
         Batch normalization of 3D tensor x
        '''

        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self,x,adj):
        """

        :param x: fea matrix emedded from Graph Conv
        :param adj: adj ,matrix of shape [batch size,l layer nodes, l+1 layer nodes]
        :return:
        """

        subs_fea,subd_adj,coarsen_adj,assign_hard = self.assignment(x,adj)

        sub_emb_fea = self.subgraph_layers(subs_fea,subd_adj)

        # coarsen_fea = torch.cat(sub_emb_fea,dim=1)

        sub_emb_fea = torch.sum(sub_emb_fea,dim=1)
        sub_emb_fea =  self.sub_act(sub_emb_fea)
        sub_emb_fea = self.apply_bn(sub_emb_fea)

        # sub_emb_fea = torch.nan_to_num(sub_emb_fea)

        sub_emb_fea = assign_hard.permute(1,0) @ sub_emb_fea

        return sub_emb_fea,coarsen_adj



class SubgraphGCN_Conv(nn.Module):
    def __init__(self,input_graph_size,subgraph_num,input_dim,output_dim,normalize_embedding=True,
            dropout=0.0, bias=False, add_self=True):
        super(SubgraphGCN_Conv,self).__init__()
        self.subgraph_num = subgraph_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.bias = bias
        self.add_self = add_self
        self.normalize_embedding = normalize_embedding

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.weight = nn.Parameter(torch.FloatTensor(subgraph_num,input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(input_graph_size,output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        """

        :param x:   subgraph_num x nodes x input_dim
        :param adj:  subgraph_num x nodes x nodes
        :return:
        """
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x


        # print(self.weight.size())
        y = torch.matmul(y, self.weight)

        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        return y




