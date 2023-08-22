import pandas as pd
import datetime
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
from tqdm import tqdm
import time

from sklearn.model_selection import KFold

import networkx as nx

import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl import DropEdge

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("..")    

from WTHMDA.utility.evaluation2 import compute_auc


    



def train_f(model, train_g, node_features, external_parameters, train_sample, label, num_epoch, criterion):
    auc_list = []
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.01)
    
    
    transform = DropEdge(p=0.2)
    
    for epoch in range(num_epoch):
        
        
        
        pred = model(train_g, node_features, external_parameters, train_sample)  
        optimizer.zero_grad()
        loss = criterion(pred.squeeze(), label.to(torch.float32))
        loss.backward()
        optimizer.step()

        
        train_auc = compute_auc(label, pred.detach())
        auc_list.append(train_auc)        
        if epoch % 30 == 0:
            print('In epoch {}, loss: {}'.format(epoch, loss))

    return auc_list

def valid_f(model, valid_g, node_features, external_parameters, valid_sample, valid_label):
    
    valid_pred = model(valid_g, node_features, external_parameters, valid_sample)
    valid_auc = compute_auc(valid_label,valid_pred.detach())
    print('验证auc',valid_auc[0:2])
    return valid_auc, valid_label, valid_pred.detach()


