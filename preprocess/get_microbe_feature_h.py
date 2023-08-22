#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  
import numpy as np
import os
import random
import networkx as nx
from gensim.models import Word2Vec
from node2vec import Node2Vec
import sys
sys.path.append("..")



def get_microbe_feature_h(M_feat_path):
    # In[ ]:
    print('get_disease_feature_h start')
    def load_weighted_Graph(fileName):
            G = nx.read_weighted_edgelist(fileName, create_using=nx.DiGraph(), nodetype=None)
            return G
    #M_feat = pd.read_csv('data/HMDAD/microbe_I_S.csv', header=None)
    M_feat = pd.read_csv(M_feat_path, header=None)
    edge_init = pd.DataFrame(data = {'micro_A' : np.zeros(M_feat.size), 'micro_B' : np.zeros(M_feat.size), 'weight' : np.zeros(M_feat.size)})
    for i in range(M_feat.shape[0]):
        for j in range(M_feat.shape[0]):
            edge_init.iloc[(i*M_feat.shape[0])+j][0] = i
            edge_init.iloc[(i*M_feat.shape[0])+j][1] = j
    for i in range(len(edge_init)):
        M_a = edge_init.iloc[i][0]
        M_b = edge_init.iloc[i][1]
        edge_init.iloc[i][2] = M_feat.iloc[int(M_a)][int(M_b)]
    edge_init.to_csv('data/HMDAD/microbe_func_edge.csv', index=False)
    dataW = pd.read_csv('data/HMDAD/microbe_func_edge.csv')
    with open('data/HMDAD/microbe_func_edge.txt', 'w+', encoding = 'utf_8') as f:
        for line in dataW.values:
            f.write((str(int(line[0])) + '\t' + str(int(line[1])) + '\t' + str(line[2]) + '\n'))


    # In[ ]:


    edge_file = 'data/HMDAD/microbe_func_edge.txt'
    G = load_weighted_Graph(edge_file)
    def w_deep_walk(G, num_paths, path_length, alpha):
        nodes = list(G.nodes())
        transitionMatrix = getTransitionMatrix(G, nodes)
        print(transitionMatrix)
        sentenceList = []
        for i in range(0, len(nodes)):
            for j in range(0, num_paths):
                indexList = generateSequence(i, transitionMatrix, path_length, alpha)
                sentence = [int(nodes[tmp]) for tmp in indexList]
                sentenceList.append(sentence)
        return sentenceList
    def getTransitionMatrix(network, nodes):
        matrix = np.zeros([len(nodes), len(nodes)])
        for i in range(0, len(nodes)):
            neighs = network.neighbors(nodes[i])
            sums = 0
            neighs_list = list(neighs)     
            for neigh in neighs_list:
                sums += network[nodes[i]][neigh]['weight']
            for j in range(0, len(nodes)):
                if i == j:
                    matrix[i, j] = 0
                else:
                    if nodes[j] not in neighs_list:
                        matrix[i, j] = 0
                    else:
                        matrix[i, j] = network[nodes[i]][nodes[j]]['weight'] / sums
        return matrix
    def generateSequence(startIndex, transitionMatrix, path_length, alpha):
        result = [startIndex]
        current = startIndex
        for i in range(0, path_length):
            if random.random() < alpha:
                nextIndex = startIndex
            else:
                probs = transitionMatrix[current]
                probs /= probs.sum()
                nextIndex = np.random.choice(len(probs), 1, p=probs)[0]
            result.append(nextIndex)
            current = nextIndex
        return result
    walks = w_deep_walk(G, num_paths = 1, path_length = 30, alpha=0)


    # In[ ]:


    def get_embeddings(w2v_model, graph):
        count = 0
        invalid_word = []
        _embeddings = {}
        for word in graph.nodes():
            word = int(word)
            if word in w2v_model.wv:
                _embeddings[word] = w2v_model.wv[word]
            else:
                invalid_word.append(word)
                count += 1
        return _embeddings
    kwargs = {'sentences':walks, 'min_count':0, 'vector_size':128, 'sg':1, 'hs':0, 'workers':3, 'window':5, 'epochs':3}
    model = Word2Vec(**kwargs)
    embeddings = get_embeddings(model, G)
    wdeepwalk128 = pd.DataFrame(np.zeros([M_feat.shape[0], 128]))
    for i in range(len(wdeepwalk128)):
        if embeddings.__contains__(i):
            wdeepwalk128.iloc[i][:] = list(embeddings[i])
    wdeepwalk128.to_csv('data/HMDAD/microbe_feature.csv', index=False)
    print('get_disease_feature_h done')

if __name__ == "__main__":
    M_feat_path = 'data/HMDAD/microbe_I_S.csv'
    get_microbe_feature_h(M_feat_path)
    print('get_disease_feature_h done')