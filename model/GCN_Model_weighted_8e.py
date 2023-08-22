import pandas as pd
import datetime
import networkx as nx
import random
from sklearn.model_selection import KFold
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")  
from WTHMDA.model.GraphConv import GraphConv
from WTHMDA.model.Weight_HeteroGraphConv import HeteroGraphConv


class RGCN(nn.Module):
    def __init__(self, in_feats_num, hid_feats1_num, out_feats_num, rel_names):
        super().__init__()

        
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats_num, hid_feats1_num, num_heads=3)
            for rel in rel_names}, aggregate='sum')
        
        self.conv1 = HeteroGraphConv({
            'interacts': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'JC_EDGE': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'LC_EDGE': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'WP_EDGE': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'Lin_EDGE': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'influence': GraphConv(in_feats_num, hid_feats1_num, weight=True, norm='none'),
            'evolve': GraphConv(in_feats_num, hid_feats1_num, weight=True),
            'relate': GraphConv(in_feats_num, hid_feats1_num, weight=True),
            'relate-by': GraphConv(in_feats_num, hid_feats1_num, weight=True)
            }, aggregate='sum')
        self.conv2 = HeteroGraphConv({

            'interacts': GraphConv(hid_feats1_num, out_feats_num),
            'JC_EDGE': GraphConv(hid_feats1_num, out_feats_num),
            'LC_EDGE': GraphConv(hid_feats1_num, out_feats_num),
            'WP_EDGE': GraphConv(hid_feats1_num, out_feats_num),
            'Lin_EDGE': GraphConv(hid_feats1_num, out_feats_num),
            'influence': GraphConv(hid_feats1_num, out_feats_num),
            'evolve': GraphConv(hid_feats1_num, out_feats_num, weight=True),
            'relate': GraphConv(hid_feats1_num, out_feats_num, weight=True),
            'relate-by': GraphConv(hid_feats1_num, out_feats_num, weight=True)
            }, aggregate='sum')

    def forward(self, graph, inputs, external_parameters):
        
        
        h = self.conv1(graph, inputs, None, external_parameters)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        h = self.conv2(graph, h, None, external_parameters)
        h = {k: F.leaky_relu(v) for k, v in h.items()}
        return h    
    
    
    
    
    
class Model(nn.Module):
    def __init__(self, in_feats_num, hid_feats1_num, out_feats_num, rel_names):    
        super().__init__()
              
        self.layers=RGCN(in_feats_num, hid_feats1_num, out_feats_num, rel_names)        
        self.seq = nn.Sequential(
            nn.Linear(out_feats_num * 2, 8),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4, 1)           
        )
        self.predict = nn.Sigmoid()
    def forward(self, g, h, external_parameters, nodes):
        

        
        h = self.layers(g, h, external_parameters)
        for i in range(len(nodes)):
            compose = torch.cat((h['disease'][nodes[i][0]], h['microbe'][nodes[i][1]]),0)
            
            if i==0:
                input_ = compose.unsqueeze(0)
            else:
                input_ = torch.cat((input_, compose.unsqueeze(0)), 0)        
        out = self.seq(input_)
        out = self.predict(out)
        
        return out