#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : local_gcn.py
@Author  : 63162
@Time    : 2024/1/25 15:42
"""

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import math


from torch_geometric.nn import GCNConv, GINConv
from subgraph import SubGraphs,SubgraphGCN_Conv
# from model.pred import PredModel
# from .utils import KMeans

class GraphConv(nn.Module):
    def __init__(self,input_dim,output_dim,add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)

        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=1, dim=1)

        return y


class MultiLayerGCN(nn.Module):
    def __init__(self,input_dim,first_dim,hidden_dim, embedding_dim,
            pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None):
        super(MultiLayerGCN, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn

        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = 0.0

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, first_dim, hidden_dim, embedding_dim,add_self, normalize=True, dropout=dropout)
        self.autoencoder = nn.ModuleList([self.conv_first, self.conv_block, self.conv_last])
        self.conv_encoder_array = nn.ModuleList([self.conv_first, self.conv_block, self.conv_last])

        self.conv_defirst, self.conv_deblock, self.conv_delast = self.build_conv_layers(
            embedding_dim, hidden_dim, first_dim, input_dim ,add_self, normalize=True, dropout=dropout)
        self.autodecoder = nn.ModuleList([self.conv_defirst, self.conv_deblock, self.conv_delast])
        self.conv_decoder_array = nn.ModuleList([self.conv_defirst, self.conv_deblock, self.conv_delast])

        self.act = nn.ReLU()


        if concat:
            self.pred_input_dim = first_dim + hidden_dim + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.first_dim = first_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim

        # self.pred_model = PredModel(self.pred_input_dim,pred_hidden_dims,label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim,first_dim, hidden_dim, embedding_dim, add_self,
            normalize=False, dropout=0.0):

        conv_first = GraphConv(input_dim=input_dim, output_dim=first_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block =GraphConv(input_dim=first_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias)

        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)

        return conv_first, conv_block, conv_last

    def build_conv_array(self,input_dim, output_dim,unit_num,add_self,
                         normalize=False, dropout=0.0):
        conv_array = []
        for i in range(len(unit_num)):
            conv_unit = GraphConv(input_dim=input_dim,output_dim=output_dim,
                                  add_self=add_self,normalize_embedding=normalize,dropout=dropout,bias=self.bias)
            conv_array.append(conv_unit)

        return conv_array



    def construct_mask(self, max_nodes, batch_num_nodes):

        '''
        For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks

        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):

        '''
         Batch normalization of 3D tensor x
        '''

        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_block, embedding_mask=None):

        '''
        Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_block(x,adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)


        x_tensor = x
        if embedding_mask is not None:

            x_tensor = x_tensor * embedding_mask

        return x_tensor



class HCGAE(MultiLayerGCN):
    def __init__(self, max_num_nodes, input_dim,first_dim, hidden_dim, embedding_dim,
                 assign_ratio=0.25, assign_num_layers=-1, num_pooling=3,
                 pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None,local=False):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(HCGAE, self).__init__(input_dim,first_dim,hidden_dim, embedding_dim,
                                                    pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    dropout=dropout,args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.local = local
        self.dropout = dropout

        self.assign_dims = []

        assign_dim = int(max_num_nodes)
        self.assign_pred_modules = nn.ModuleList()
        self.assign_layers = nn.ModuleList()

        self.embed_dim_list = [input_dim] + [first_dim,hidden_dim, embedding_dim]
        self.input_node_size = max_num_nodes
        self.assign_ratio = assign_ratio
        self.assign_node_num = [int(math.pow(self.assign_ratio, i) * self.input_node_size) for i in
                                range(self.num_pooling + 1)]


        for i in range(self.num_pooling):
            self.assign_layers.append(SubGraphs(self.assign_node_num[i], self.assign_node_num[i + 1],
                                                self.embed_dim_list[i + 1], self.embed_dim_list[i + 1], args.batch_size))

        # differ local and global
        node_num_list = []
        for i in range(num_pooling):
            node_num_list.append(assign_dim)
            assign_dim = int(assign_dim * assign_ratio)

        if self.local != True:
            self.build_global_gcn(input_dim, first_dim, hidden_dim, node_num_list, add_self)

        for m in self.modules():
            if isinstance(m, GraphConv) or isinstance(m, SubgraphGCN_Conv):

                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)


    def build_global_gcn(self,input_dim,first_dim,hidden_dim,node_num_list, add_self,normalize=False):

        assign_input_dim, assign_first_dim, assign_hidden_dim = node_num_list
        self.global_assign_conv_block_modules = nn.ModuleList()
        assign_conv_first_block = GraphConv(input_dim=first_dim, output_dim=assign_first_dim,
                                      add_self=add_self,normalize_embedding=normalize, bias=self.bias)
        assign_conv_last_block = GraphConv(input_dim=hidden_dim, output_dim=assign_hidden_dim,
                                            add_self=add_self, normalize_embedding=normalize, bias=self.bias)
        self.global_assign_conv_block_modules.append(assign_conv_first_block)
        self.global_assign_conv_block_modules.append(assign_conv_last_block)

        self.global_assign_conv_beblock_modules = nn.ModuleList()
        assign_conv_first_beblock = GraphConv(input_dim=hidden_dim, output_dim=assign_first_dim,
                                            add_self=add_self, normalize_embedding=normalize, bias=self.bias)
        assign_conv_last_beblock = GraphConv(input_dim=first_dim, output_dim=assign_input_dim,
                                           add_self=add_self, normalize_embedding=normalize, bias=self.bias)
        self.global_assign_conv_beblock_modules.append(assign_conv_first_beblock)
        self.global_assign_conv_beblock_modules.append(assign_conv_last_beblock)

    def build_part(self,args,node_num_list,dim_list):
        layer_num = len(node_num_list)
        if args.partition_classifier == "softmax":
            self.partitioner = "softmax"
            self.subgraph_partiter = [nn.Linear(layer_num[i], layer_num[i+1]) for i in range(layer_num-1)]

        # elif args.partition_classifier == "k-means":
        #     self.partitioner = "softmax"
        #     self.subgraph_partiter = [KMeans(layer_num[i+1],dim_list[i]) for i in range(layer_num-1)]


    def partition(self,embedding_matrix,assign_matrix,deepth):

        if self.partitioner == "k-means":
            partition_logits = F.softmax(embedding_matrix, dim=1)
            # distances = [torch.norm(x - x_train) for x_train in self.X_train]

        elif self.partitioner == "softmax":

            # emb_input = torch.matmul(embedding_matrix, embedding_matrix.permute(0,2,1))
            emb_output = F.softmax(self.subgraph_partiter[deepth](assign_matrix),dim=1)
            emb_mapping = torch.argmax(emb_output, dim=2)
            batch_size = emb_output.size(0)
            input_dim = assign_matrix.size(1)
            output_dim = emb_output.size(1)
            hard_mapping = torch.zeros(batch_size,input_dim,output_dim)
            for i in range(batch_size):
                for j in range(input_dim):
                    hard_mapping[i][j][emb_mapping[i][j]] = 1

            return hard_mapping

    def label_tracer(self):
        print("")

    def forward(self, x, adj):

        x_hidden = []
        x_a = x
        # mask

        self.fea_store = [x_a]
        self.adj_store = [adj]

        for i in range(self.num_pooling - 1):

            embedding_tensor = self.gcn_forward(x_a, adj, self.conv_encoder_array[i], embedding_mask=None)

            self.fea_store.append(embedding_tensor)

            x = embedding_tensor

            if self.local == True:
                x, adj = self.assign_layers[i](x, adj)
            else:
                assign_tensor = self.gcn_forward(embedding_tensor, adj, self.global_assign_conv_block_modules[i],
                                                 embedding_mask=None)
                assign_tensor = nn.Softmax(dim=-1)(assign_tensor)
                x = torch.matmul(torch.transpose(assign_tensor, 0, 1), embedding_tensor)
                adj = torch.matmul(torch.transpose(assign_tensor, 0, 1), adj)
                adj = torch.matmul(adj,assign_tensor)

            self.adj_store.append(adj)

            x_a = x

        x_hidden.append(self.fea_store[1])
        x_a = self.gcn_forward(x_a,adj,self.conv_encoder_array[self.num_pooling-1],embedding_mask=None)

        for i in range(self.num_pooling - 1):

            embedding_mask = None

            embedding_tensor = self.gcn_forward(x_a, adj, self.conv_decoder_array[i], embedding_mask)


            assign_tensor = self.gcn_forward(embedding_tensor, adj, self.global_assign_conv_beblock_modules[i],
                                             embedding_mask)
            assign_tensor = nn.Softmax(dim=-1)(assign_tensor)

            # update  features and adj matrix
            x =  torch.matmul(torch.transpose(assign_tensor, 0, 1), embedding_tensor)
            adj =  torch.matmul(torch.transpose(assign_tensor, 0, 1), adj)
            adj =  torch.matmul(adj, assign_tensor)


            self.adj_store.append(adj)
            self.fea_store.append(x)

            x_a = x

        x_hidden.append(self.fea_store[-1])

        return x_hidden

    def js_loss(self,adj_en, adj_de):

        vec_en = F.normalize(torch.sum(adj_en,dim=1),dim=0)
        vec_de = F.normalize(torch.sum(adj_de,dim=1),dim=0)
        vec_en = F.softmax(vec_en,dim=0)
        vec_de = F.softmax(vec_de,dim=0)
        middle = F.softmax(0.5 * vec_de + 0.5 * vec_en, dim=0)
        kl_en = vec_en*torch.log(vec_en/middle)
        kl_de = vec_de*torch.log(vec_de/middle)
        kl = 0.5 * kl_en + 0.5 * kl_de
        kl = torch.sum(kl)
        # print(kl)
        return kl

    def unsupervised_loss(self):

        fea_compared_size = len(self.fea_store) // 2
        adj_compared_size = len(self.adj_store) // 2

        loss = torch.zeros(1).cuda()
        for i in range(fea_compared_size):
            loss += self.compared_loss(self.fea_store[i],self.fea_store[-1-i])
        for i in range(adj_compared_size):
            loss += self.compared_loss(self.adj_store[i],self.adj_store[-1-i])
        # js_loss = self.js_loss(self.adj_store[0], self.adj_store[-1])

        return loss
    def compared_loss(self,original_matrix,compressed_matrix):
        # print("Unsupervised Loss!")
        batch_size = original_matrix.shape[0]
        assert batch_size!=0
        distribution = 'bernoulli'
    #   bernoulli, gaussian
        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(original_matrix, compressed_matrix, reduction="mean").div(batch_size)

        elif distribution == 'gaussian':
            compressed_matrix = F.sigmoid(compressed_matrix)
            recon_loss = F.mse_loss(original_matrix, compressed_matrix, reduction="mean").div(batch_size)

        else:
            recon_loss = None

        return recon_loss



    def loss(self):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7


        loss = self.compared_loss(self.fea_store[1],self.fea_store[-1])

        js_loss = self.js_loss(self.adj_store[0],self.adj_store[-1])

        loss += js_loss

        return loss




