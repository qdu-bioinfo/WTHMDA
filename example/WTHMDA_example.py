#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import random
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
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
sys.path.append("../.")    
from WTHMDA2.utility.dataset_weighted_8e import Dataset
#from model.GraphConv import EdgeWeightNorm
from WTHMDA2.model.GCN_Model_weighted_8e import RGCN
from WTHMDA2.model.GCN_Model_weighted_8e import Model
from WTHMDA2.model.run_f_weighted import train_f
from WTHMDA2.model.run_f_weighted import valid_f
from WTHMDA2.utility.evaluation2 import save_plot_auc
from WTHMDA2.utility.evaluation2 import plot_auc
from WTHMDA2.utility.evaluation2 import plot_prc_curve


# In[2]:
def WTHMDA_example(train_edge_path, test_edge_path, save_path):

    #dm_path = '../data/HMDAD/microbe_disease_edge.csv'
    #dm = pd.read_csv('data/HMDAD/microbe_disease_edge.csv')

    mm_path = 'data/example/microbe_func_edge.csv'
    dd_path = 'data/example/disease_func_edge.csv'
    jc_path = 'data/example/JC_edge.csv'
    lc_path = 'data/example/LC_edge.csv'
    wp_path = 'data/example/WP_edge.csv'
    lin_path = 'data/example/Lin_edge.csv'
    rank_path = 'data/example/tree_edge.csv'
    disease_features_path = 'data/example/disease_feature.csv'
    microbe_features_path = 'data/example/microbe_feature.csv'
    etype = ('disease', 'relate', 'microbe')
    etype2 = ('microbe', 'relate-by', 'disease')
    n2p_rate = 1
    #name = 'result/example'
    name = save_path


    # In[3]:


    #train_edge = pd.read_csv('data/example/train_edge.csv')
    #test_edge = pd.read_csv('data/example/test_edge.csv')
    train_edge = pd.read_csv(train_edge_path)
    test_edge = pd.read_csv(test_edge_path)
    train_edge_1 = train_edge[train_edge['label']==1]
    test_edge_1 = test_edge[test_edge['label']==1]
    dm = pd.concat([train_edge_1,test_edge_1], axis=0)
    dm = dm.drop(['edge_id', 'label'], axis=1)
    dm.columns = ['disease','microbe']


    # In[4]:


    g_data = Dataset(dm, mm_path, dd_path, rank_path, disease_features_path, microbe_features_path)
    dmhg = g_data.build_graph()
    m_size = dmhg.num_nodes('microbe')
    d_size = dmhg.num_nodes('disease')


    # In[5]:


    mm = pd.read_csv(mm_path)
    dd = pd.read_csv(dd_path)
    jc = pd.read_csv(jc_path)
    lc = pd.read_csv(lc_path)
    wp = pd.read_csv(wp_path)
    lin = pd.read_csv(lin_path)
    mm_w = torch.Tensor(mm['weight'].values)
    mm_w = mm_w.reshape(m_size*m_size, 1)
    jc_w = torch.Tensor(jc['JC'].values)
    jc_w = jc_w.reshape(m_size*m_size, 1)
    lc_w = torch.Tensor(lc['LC'].values)
    lc_w = lc_w.reshape(m_size*m_size, 1)
    wp_w = torch.Tensor(wp['WP'].values)
    wp_w = wp_w.reshape(m_size*m_size, 1)
    lin_w = torch.Tensor(lin['Lin'].values)
    lin_w = lin_w.reshape(m_size*m_size, 1)
    dd_w = torch.Tensor(dd['weight'].values)
    dd_w = dd_w.reshape(d_size*d_size, 1)
    dmhg.edges['interacts'].data['w'] = mm_w
    dmhg.edges['JC_EDGE'].data['w'] = jc_w
    dmhg.edges['LC_EDGE'].data['w'] = lc_w
    dmhg.edges['WP_EDGE'].data['w'] = wp_w
    dmhg.edges['Lin_EDGE'].data['w'] = lin_w
    dmhg.edges['influence'].data['w'] = dd_w


    # In[6]:


    external_parameters = {'interacts':{'edge_weight':mm_w},
                        'JC_EDGE':{'edge_weight':jc_w},
                        'LC_EDGE':{'edge_weight':lc_w},
                        'WP_EDGE':{'edge_weight':wp_w},
                        'Lin_EDGE':{'edge_weight':lin_w},
                        'influence':{'edge_weight':dd_w}}


    # In[7]:


    dm_no_edges = g_data.create_no_edges()
    eids = np.arange(dmhg.number_of_edges(etype = etype))  
    kf_fake_edges = g_data.select_fake_edges(n2p_rate)
    kf_eids_array = g_data.get_eids()
    D_id = dmhg.edges(etype='relate')[0].numpy()


    # In[8]:


    train_index = dmhg.edge_ids(train_edge_1.node1.tolist(), train_edge_1.node2.tolist(), etype = etype)
    valid_index = dmhg.edge_ids(test_edge_1.node1.tolist(), test_edge_1.node2.tolist(), etype = etype)


    # In[11]:


    train_sample = list(zip(train_edge.node1.tolist(), train_edge.node2.tolist()))
    valid_sample = list(zip(test_edge.node1.tolist(), test_edge.node2.tolist()))


    # In[12]:


    valid_auc_list = []
    valid_prf_list = []
    valid_prf_all_list = []
    tprs=[]
    aucs=[]
    mean_fpr=np.linspace(0,1,100)
    i_r = 0
    #times = 0
    mean_precision = 0.0
    mean_recall = np.linspace(0,1,100)
    mean_average_precision = []
    vd_data_list = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    kf.get_n_splits(kf_eids_array, D_id)
    #fig = plt.figure(figsize=(6,4), dpi=100)
    #for train_index, valid_index in kf.split(kf_eids_array, D_id):
    train_auc_list = []
    #print('----------第', times + 1, '次----------')
    #times += 1
    train_g, valid_g = g_data.build_g(valid_index, train_index)
    #train_sample, valid_sample = g_data.get_sample(train_index, valid_index, n2p_rate)
    label = torch.tensor(train_edge.label)
    valid_label = torch.tensor(test_edge.label)
    node_features = g_data.get_feats()



    model = Model(in_feats_num = 128, hid_feats1_num = 64, out_feats_num = 16, rel_names = dmhg.etypes)
    train_auc = train_f(model, train_g, node_features, external_parameters, train_sample, label, 150, nn.BCELoss())
    train_auc_list = train_auc_list + train_auc
    train_auc_df = pd.DataFrame(train_auc_list)
    model.eval()
    with torch.no_grad():
        valid_auc, v_label, v_pred = valid_f(model, train_g, node_features, external_parameters, valid_sample, valid_label)
        valid_auc_list.append(valid_auc) 
        vd_l2p = torch.cat((v_label.unsqueeze(1), v_pred), 1)
        if i_r==0:
            result_pack = vd_l2p
        else:
            result_pack = torch.cat((result_pack, vd_l2p), axis=0)        
        #tprs, aucs, i_r = save_plot_auc(v_label, v_pred, mean_fpr, tprs, aucs, i_r)
        vd_npl2p = vd_l2p.detach().numpy()
        vd_data_list.append(vd_npl2p)                
    valid_auc_df = pd.DataFrame(valid_auc_list)
    print('valid : 本次预测各指标均值', list(valid_auc_df.mean()), '结果已保存' )
    #plot_auc(tprs, aucs, mean_fpr, name)
    #mean_auprc, auprcs = plot_prc_curve(vd_data_list, name)
    result_df = pd.DataFrame(result_pack.numpy())
    result_df.columns = ['label','pred']
    result_path = name + '_result.csv'
    result_df.to_csv(result_path, index=False)


# In[ ]:
if __name__ == "__main__":

    train_edge_path = 'data/example/train_edge.csv'
    test_edge_path = 'data/example/test_edge.csv'
    save_path = 'result/example'
    WTHMDA_example(train_edge_path, test_edge_path, save_path)
    print('WTHMDA_example done')




# In[ ]:




