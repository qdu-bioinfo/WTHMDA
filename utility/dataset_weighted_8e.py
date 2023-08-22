import pandas as pd
import datetime
import random
import numpy as np
import math
import copy
import scipy.sparse as sp
from collections import Counter


import networkx as nx
import dgl
import dgl.nn as dglnn
import dgl.function as fn

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("..")    


class Dataset(object):
    def __init__(self, dm, mm_path, dd_path, rank_path, disease_features_path, microbe_features_path):
        self.dm = dm
        self.mm = pd.read_csv(mm_path)
        self.dd = pd.read_csv(dd_path)
        self.rankT = pd.read_csv(rank_path)
        self.disease_featuresDF = pd.read_csv(disease_features_path)
        self.microbe_featuresDF = pd.read_csv(microbe_features_path)

        self.etype = ('disease', 'relate', 'microbe')
        self.etype2 = ('microbe', 'relate-by', 'disease')

    def build_graph(self):

        self.dm.columns = ['disease','microbe']
        disease = self.dm['disease'].tolist()   
        self.disease = disease
        microbe = self.dm['microbe'].tolist()  
        self.microbe = microbe
        
        m1 = self.mm['micro_A'].tolist()         
        m2 = self.mm['micro_B'].tolist()
        
        

        d1 = self.dd['disease_A'].tolist()       
        d2 = self.dd['disease_B'].tolist()
        
        h = self.rankT['higher'].tolist()
        l = self.rankT['lower'].tolist()
        
        mm_graph_data = (m1, m2)
        dd_graph_data = (d1, d1)
        graph_data = {
                 ('microbe', 'interacts', 'microbe'): (m1, m2),
                ('microbe', 'JC_EDGE', 'microbe'): (m1, m2),
                ('microbe', 'LC_EDGE', 'microbe'): (m1, m2),
                ('microbe', 'WP_EDGE', 'microbe'): (m1, m2),
                ('microbe', 'Lin_EDGE', 'microbe'): (m1, m2),            
                ('disease', 'relate', 'microbe'): (disease, microbe),
                ('microbe', 'relate-by', 'disease'): (microbe, disease),
                ('microbe', 'evolve', 'microbe'): (h, l),
                ('disease', 'influence', 'disease'): (d1, d2)
        }
        

        dmhg = dgl.heterograph(graph_data)
        
        microbe_featuresDF = self.microbe_featuresDF
        disease_featuresDF = self.disease_featuresDF

        

        dmhg.nodes['microbe'].data['feature'] = torch.Tensor(microbe_featuresDF.values)
        dmhg.nodes['disease'].data['feature'] = torch.Tensor(disease_featuresDF.values) 

        self.dmhg = dmhg
        return dmhg
    
    def get_feats(self):
        
        microbe_feats = self.dmhg.nodes['microbe'].data['feature']
        microbe_feats2 = self.dmhg.nodes['microbe'].data['feature']
        disease_feats = self.dmhg.nodes['disease'].data['feature']
        disease_feats2 = self.dmhg.nodes['disease'].data['feature']
        
        node_features = {'disease':disease_feats, 'microbe':microbe_feats, 'microbe2':microbe_feats2, 'disease2':disease_feats2}
        self.node_features = node_features
        
        return node_features
    
    def get_eids(self):
        kf_eids_array = np.arange(self.dmhg.number_of_edges(etype=self.etype))
        self.kf_eids_array = kf_eids_array
        return kf_eids_array
    
    def create_no_edges(self):
        dm_edges = list(zip(self.disease, self.microbe))
        dl = np.arange(self.dmhg.num_nodes('disease'))
        listd = dl.tolist()
        ml = np.arange(self.dmhg.num_nodes('microbe'))
        listm = ml.tolist()
        dm_no_edges = [(i, j) for i in listd for j in listm if (i, j) not in dm_edges]
        self.dm_no_edges = dm_no_edges
        return dm_no_edges
    
    def select_fake_edges(self, n2p_rate):
        d, m = self.dmhg.edges(etype = self.etype)
        real_edges = list(zip(d.tolist(), m.tolist()))
        fake_edges = random.sample(self.dm_no_edges, len(real_edges) * n2p_rate)
        self.fake_edges = fake_edges
        samples = real_edges + fake_edges
        return fake_edges
    
    def select_fake_edges2(self, train_d, valid_d, train_len, valid_len, dm_no_edges, n2p_rate):
        train_fe_pool = copy.deepcopy(dm_no_edges)
        valid_fe_pool = copy.deepcopy(dm_no_edges)
        for i in reversed(train_fe_pool):
            if i[0] in valid_d:
                train_fe_pool.remove(i)   
        for j in reversed(valid_fe_pool):
            if j[0] in train_d:
                valid_fe_pool.remove(j)
        train_fake_edges = random.sample(train_fe_pool, train_len * n2p_rate)
        valid_fake_edges = random.sample(valid_fe_pool, valid_len * n2p_rate)
        return train_fake_edges, valid_fake_edges
    
    def select_fake_edges3(self, train_d, valid_d, train_len, valid_len, dm_no_edges, n2p_rate):
        train_fe_pool = copy.deepcopy(dm_no_edges)
        valid_fe_pool = copy.deepcopy(dm_no_edges)
        for i in reversed(train_fe_pool):
            if i[1] in valid_d:
                train_fe_pool.remove(i)   
        for j in reversed(valid_fe_pool):
            if j[1] in train_d:
                valid_fe_pool.remove(j)
        train_fake_edges = random.sample(train_fe_pool, train_len * n2p_rate)
        valid_fake_edges = random.sample(valid_fe_pool, valid_len * n2p_rate)
        return train_fake_edges, valid_fake_edges
    
    def build_g(self,v_index, t_index):
        
        d, m = self.dmhg.edges(etype=self.etype)
        train_g = dgl.remove_edges(self.dmhg, self.kf_eids_array[v_index], etype=self.etype)
        t_dpeids = self.dmhg.edge_ids(m[v_index], d[v_index], etype = self.etype2)
        train_g = dgl.remove_edges(train_g, t_dpeids, etype=self.etype2)
        self.train_g = train_g
        valid_g = dgl.remove_edges(self.dmhg, self.kf_eids_array[t_index], etype=self.etype)
        v_dpeids = self.dmhg.edge_ids(m[t_index], d[t_index], etype = self.etype2)
        valid_g = dgl.remove_edges(valid_g, v_dpeids, etype=self.etype2)
        self.valid_g = valid_g
        return train_g, valid_g
    
    def get_sample2(self, train_index, valid_index, train_fake_edges, valid_fake_edges, n2p_rate):
        
        t_d, t_m = self.train_g.edges(etype = self.etype)
        train_real_edges = list(zip(t_d.tolist(), t_m.tolist()))
        train_samples = train_real_edges + train_fake_edges
        self.train_samples = train_samples
        self.train_real_edges = train_real_edges
        v_d, v_m = self.valid_g.edges(etype = self.etype)
        valid_real_edges = list(zip(v_d.tolist(), v_m.tolist()))
        valid_samples = valid_real_edges + valid_fake_edges
        random.shuffle(valid_samples)
        random.shuffle(train_samples)
        self.valid_samples = valid_samples
        self.valid_real_edges = valid_real_edges
        return train_samples, valid_samples
    
    def get_sample(self, train_index, valid_index, n2p_rate):
        
        t_d, t_m = self.train_g.edges(etype = self.etype)
        train_real_edges = list(zip(t_d.tolist(), t_m.tolist()))
        train_index = list(train_index)
        sample_len = len(self.kf_eids_array)
        out_index = []
        for i in range(n2p_rate):
            for num in train_index:
                out_index.append(sample_len * i + num)
        train_fake_edges = []
        for i in out_index:
            train_fake_edges.append(self.fake_edges[i])
        train_samples = train_real_edges + train_fake_edges
        self.train_samples = train_samples
        self.train_real_edges = train_real_edges

        valid_no_edges = copy.deepcopy(self.fake_edges)
        for i in train_fake_edges:
            valid_no_edges.remove(i)
        v_d, v_m = self.valid_g.edges(etype = self.etype)
        valid_real_edges = list(zip(v_d.tolist(), v_m.tolist()))
        valid_fake_edges = valid_no_edges
        valid_samples = valid_real_edges + valid_fake_edges
        self.valid_samples = valid_samples
        self.valid_real_edges = valid_real_edges
        
        return train_samples, valid_samples
    
    def get_label(self):
        train_label = []
        valid_label = []
        for i in range(len(self.train_samples)):
            if self.train_samples[i] in self.train_real_edges:
                train_label.append(1)
            else:
                train_label.append(0)
        train_label = torch.tensor(train_label)
        self.train_label = train_label
        for j in range(len(self.valid_samples)):
            if self.valid_samples[j] in self.valid_real_edges:
                valid_label.append(1)
            else:
                valid_label.append(0)
        valid_label = torch.tensor(valid_label)
        self.valid_label = valid_label
        
        return train_label, valid_label
    
   