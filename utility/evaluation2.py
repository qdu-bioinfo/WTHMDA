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
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score



def compute_auc(labels, scores):

    roc_auc = roc_auc_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    auprc = auc(recall, precision)
    return roc_auc ,auprc

def save_plot_auc(v_label, v_pred, mean_fpr, tprs, aucs, i_r):
    fpr,tpr,thresholds=roc_curve(v_label, v_pred)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.4f)'% (i_r,roc_auc))
    i_r += 1
    
    return tprs, aucs, i_r

def plot_auc(tprs, aucs, mean_fpr, name):

    plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
    mean_tpr=np.mean(tprs,axis=0)
    mean_tpr[-1]=1.0
    mean_auc=auc(mean_fpr,mean_tpr)
    std_auc=np.std(tprs,axis=0)
    
    std = np.std(aucs)
    
    plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (AUC=%0.4f $\pm$ %0.3f)'%(mean_auc, std),lw=2,alpha=.8)
    std_tpr=np.std(tprs,axis=0)
    tprs_upper=np.minimum(mean_tpr+std_tpr,1)
    tprs_lower=np.maximum(mean_tpr-std_tpr,0)
    plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right', prop = {'size':6})
    
    plot_file = name + 'plot_1.jpg'
    #plt.savefig(plot_file)
    
    plt.show()
    
def plot_prc_curve(vd_data, name, label_column=0, score_column=1):
    datasize = len(vd_data)
    precisions = []
    aucs = []
    recall_array = []
    precision_array = []
    mean_recall = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(6,4), dpi=100)
    for i in range(len(vd_data)):
        precision, recall, _ = precision_recall_curve(vd_data[i][:, label_column], vd_data[i][:, score_column])
        recall_array.append(recall)
        precision_array.append(precision)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1])[::-1])
        roc_auc = auc(recall, precision)
        aucs.append(roc_auc)
        plt.plot(recall,precision,lw=1,alpha=0.3,label='PRC fold %d(AUPRC=%0.2f)'% (i,roc_auc))
        
    
    mean_precision = np.mean(precisions, axis=0)
    mean_recall = mean_recall[::-1]
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    if datasize>1:
        plt.plot(mean_recall, mean_precision, color='blue',
                 label=r'Mean PRC (AUPRC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    if datasize>1:
        plt.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left", prop = {'size':6})
    
    plot_file = name + 'plot_2.jpg'
    #plt.savefig(plot_file)
    
    plt.show()
    return mean_auc, aucs

